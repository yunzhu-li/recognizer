//
// object-recognizer
// Copyright (C) 2017-2018 Yunzhu Li
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.
//

// VideoCaptureCoordinator handles image capture and preview using camera
import UIKit
import AVFoundation

protocol VideoFrameCaptureDelegate {
    func frameCapture(didCapture image: UIImage)
}

class VideoCaptureCoordinator: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    private var avcSession: AVCaptureSession?
    private var avcDevice: AVCaptureDevice?
    private var avcInput: AVCaptureInput?
    private var avcOutput: AVCaptureOutput?
    private var avcPreviewLayer: AVCaptureVideoPreviewLayer?
    private var avcDataOutput: AVCaptureVideoDataOutput?

    private var captureDelegate: VideoFrameCaptureDelegate
    private var _captureNextFrame = false

    public init(_ delegate: VideoFrameCaptureDelegate) {
        captureDelegate = delegate
    }

    // Handle camera authorization
    func cameraAuth(completionHandler handler: @escaping (Bool) -> Void) {
        // Request access to camera
        AVCaptureDevice.requestAccess(for: .video, completionHandler: { (granted) in
            // Execute handler in main thread
            DispatchQueue.main.async {
                handler(granted)
            }
        })
    }

    // Create and configure capture session and related resources
    func initCapture(previewView: UIView) -> String? {
        // Session
        avcSession = AVCaptureSession()
        if avcSession == nil {
            return "Could not create AVCaptureSession"
        }

        // Currently we only need 64x64 for recognition
        if avcSession!.canSetSessionPreset(.hd1280x720) {
            avcSession?.sessionPreset = .hd1280x720
        }

        // Device
        avcDevice = AVCaptureDevice.default(for: .video)
        if avcDevice == nil { return "Could not request an AVCaptureDevice" }

        // Input
        do {
            try avcInput = AVCaptureDeviceInput(device: avcDevice!)
            avcSession?.addInput(avcInput!)
        } catch {
            return "Could not create AVCaptureDeviceInput"
        }

        // Preview layer
        avcPreviewLayer = AVCaptureVideoPreviewLayer(session: avcSession!)
        if avcPreviewLayer == nil { return "Could not create AVCaptureVideoPreviewLayer" }

        avcPreviewLayer?.frame.size = previewView.frame.size
        avcPreviewLayer?.videoGravity = .resizeAspectFill
        avcPreviewLayer?.connection?.videoOrientation = .portrait // Only support portrait for now
        previewView.layer.addSublayer(avcPreviewLayer!)

        // Configure video frame collection
        avcDataOutput = AVCaptureVideoDataOutput()
        if avcDataOutput == nil { return "Could not create AVCaptureVideoDataOutput" }

        let queue = DispatchQueue(label: "net.blupig.object-recognizer")
        avcDataOutput?.setSampleBufferDelegate(self, queue: queue)

        if avcSession?.canAddOutput(avcDataOutput!) != true {
            return "Could not add AVCaptureVideoDataOutput to AVCaptureSession"
        }

        avcSession?.addOutput(avcDataOutput!)

        // No error
        return nil
    }

    // Set preview layer to same size as the UIView provided
    func syncPreviewLayerSize(previewView: UIView) {
        avcPreviewLayer?.frame.size = previewView.frame.size
    }

    // Start / stop capture session
    func turnVideoCapture(on: Bool) {
        if on {
            avcSession?.startRunning()
        } else {
            avcSession?.stopRunning()
        }
    }

    // Schedule next frame to be captured
    func captureNextFrame() {
        _captureNextFrame = true
    }

    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        if _captureNextFrame {
            _captureNextFrame = false
            // Convert to UIImage
            if let uiImage = cmBufferToImage(cmBuffer: sampleBuffer) {
                // Give to delegate in main thread
                DispatchQueue.main.async {
                    self.captureDelegate.frameCapture(didCapture: uiImage)
                }
            }
        }
    }

    // Buffer to image convenience function
    private func cmBufferToImage(cmBuffer: CMSampleBuffer) -> UIImage? {
        let cvBuffer = CMSampleBufferGetImageBuffer(cmBuffer)
        if cvBuffer == nil { return nil }
        let ciImage = CIImage(cvImageBuffer: cvBuffer!)
        let context = CIContext()
        let cgImage = context.createCGImage(ciImage, from: CVImageBufferGetCleanRect(cvBuffer!))
        if cgImage == nil { return nil }
        return UIImage(cgImage: cgImage!)
    }
}
