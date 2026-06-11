import Foundation
import ScreenCaptureKit
import AVFoundation

guard CommandLine.arguments.count > 1 else {
    fputs("Usage: AudioCapture <output.wav>\n", stderr)
    exit(1)
}

let outputURL = URL(fileURLWithPath: CommandLine.arguments[1])
// Mic file sits next to the system-audio file: last_recording_mic.wav
let micURL: URL = {
    let stem = outputURL.deletingPathExtension().path
    let ext  = outputURL.pathExtension
    return URL(fileURLWithPath: "\(stem)_mic.\(ext)")
}()

// MARK: - ScreenCaptureKit recorder (system audio output)

final class Recorder: NSObject, SCStreamOutput, SCStreamDelegate {
    private var scStream: SCStream?
    private var audioFile: AVAudioFile?
    private let ioQueue = DispatchQueue(label: "audio.capture.io", qos: .userInitiated)

    // Mic (AVAudioEngine, microphone input)
    private var micEngine: AVAudioEngine?
    private var micFile: AVAudioFile?

    func start(to url: URL) async throws {
        startMic()

        let content = try await SCShareableContent.excludingDesktopWindows(
            false, onScreenWindowsOnly: false
        )
        guard let display = content.displays.first else {
            throw NSError(domain: "AudioCapture", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "No display available"])
        }

        let filter = SCContentFilter(display: display,
                                     excludingApplications: [],
                                     exceptingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 48000
        config.channelCount = 2
        // Minimal video — we only want audio
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)

        scStream = SCStream(filter: filter, configuration: config, delegate: self)
        try scStream!.addStreamOutput(self, type: .audio, sampleHandlerQueue: ioQueue)
        try await scStream!.startCapture()

        fputs("[AudioCapture] System audio → \(url.lastPathComponent)\n", stderr)
    }

    func stop() {
        stopMic()
        let s = scStream
        scStream = nil
        Task {
            try? await s?.stopCapture()
            ioQueue.sync { self.audioFile = nil }
            fputs("[AudioCapture] Done\n", stderr)
            exit(0)
        }
    }

    // MARK: Mic helpers

    private func startMic() {
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let format = inputNode.outputFormat(forBus: 0)

        guard format.sampleRate > 0 else {
            fputs("[AudioCapture] Mic: no input device or permission denied — skipping\n", stderr)
            return
        }

        do {
            micFile = try AVAudioFile(forWriting: micURL, settings: format.settings)
        } catch {
            fputs("[AudioCapture] Mic file error: \(error) — skipping\n", stderr)
            return
        }

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
            try? self?.micFile?.write(from: buffer)
        }

        do {
            try engine.start()
            micEngine = engine
            fputs("[AudioCapture] Mic → \(micURL.lastPathComponent)\n", stderr)
        } catch {
            fputs("[AudioCapture] Mic start error: \(error) — skipping\n", stderr)
            inputNode.removeTap(onBus: 0)
            micFile = nil
        }
    }

    private func stopMic() {
        guard let engine = micEngine else { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        micEngine = nil
        micFile = nil
        fputs("[AudioCapture] Mic done\n", stderr)
    }

    // MARK: SCStreamOutput

    func stream(_ stream: SCStream,
                didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .audio,
              let formatDesc = sampleBuffer.formatDescription else { return }
        let format = AVAudioFormat(cmAudioFormatDescription: formatDesc)

        if audioFile == nil {
            audioFile = try? AVAudioFile(forWriting: outputURL, settings: format.settings)
        }
        guard let audioFile else { return }

        let frameCount = AVAudioFrameCount(sampleBuffer.numSamples)
        guard frameCount > 0,
              let pcm = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else { return }
        pcm.frameLength = frameCount

        do {
            try sampleBuffer.withAudioBufferList { abl, _ in
                let dst = UnsafeMutableAudioBufferListPointer(pcm.mutableAudioBufferList)
                for (i, src) in abl.enumerated() {
                    guard i < dst.count,
                          let srcPtr = src.mData,
                          let dstPtr = dst[i].mData else { continue }
                    memcpy(dstPtr, srcPtr, Int(src.mDataByteSize))
                }
                try audioFile.write(from: pcm)
            }
        } catch {
            fputs("[AudioCapture] Write error: \(error)\n", stderr)
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("[AudioCapture] Stream error: \(error.localizedDescription)\n", stderr)
    }
}

let recorder = Recorder()
signal(SIGINT)  { _ in recorder.stop() }
signal(SIGTERM) { _ in recorder.stop() }

Task {
    do {
        try await recorder.start(to: outputURL)
    } catch {
        fputs("[AudioCapture] Failed to start: \(error.localizedDescription)\n", stderr)
        exit(1)
    }
}

RunLoop.main.run()
