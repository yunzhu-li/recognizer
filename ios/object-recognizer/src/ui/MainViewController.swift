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

// Main Screen
import UIKit
import AVFoundation

class MainViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate, VideoFrameCaptureDelegate, UITableViewDelegate, UITableViewDataSource {

    @IBOutlet weak var viewCamera: UIView!
    @IBOutlet weak var lblCameraInfo: UILabel!
    @IBOutlet weak var viewStillImage: UIView!
    @IBOutlet weak var imgStillImage: UIImageView!
    @IBOutlet weak var tableView: UITableView!

    var vcc: VideoCaptureCoordinator?
    var cameraAvailable = false
    var realTimeCaptureEnabled = false
    var sampleTimer: Timer?
    var annotations: [ImageAnnotation]?

    // UIAlertController convenience function
    func alert(title: String, message: String) {
        let controller = UIAlertController(title: title, message: message, preferredStyle: .alert)
        controller.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))
        self.present(controller, animated: true, completion: nil)
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        // Create video capture coordinator
        vcc = VideoCaptureCoordinator(self)

        // Request for camera access
        vcc?.cameraAuth { (granted) in
            // Show info label if not granted
            self.lblCameraInfo.isHidden = granted

            if !granted { return }

            if let error = self.vcc?.initCapture(previewView: self.viewCamera) {
                self.alert(title: "Error", message: error)
                return
            }

            // No error, real-time mode on by default
            self.cameraAvailable = true
            self.realTimeCaptureEnabled = true
            self.realTimeCapture(on: true)
        }
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        // Sync preview layer size
        vcc?.syncPreviewLayerSize(previewView: viewCamera)

        // Start automatic capture
        realTimeCapture(on: true)
    }

    override func viewWillDisappear(_ animated: Bool) {
        // Pause real-time capture
        realTimeCapture(on: false)
    }

    // Pick image from photo library
    @IBAction func btnPhotoLibraryAct(_ sender: UIBarButtonItem) {
        if UIImagePickerController.isSourceTypeAvailable(.photoLibrary) {
            let pc = UIImagePickerController()
            // Customize
            pc.navigationBar.isTranslucent = false
            pc.navigationBar.barTintColor = UIColor.darkGray
            pc.navigationBar.tintColor = UIColor.white
            pc.navigationBar.titleTextAttributes = [NSAttributedStringKey.foregroundColor : UIColor.white]
            // Receive results
            pc.delegate = self
            pc.sourceType = .photoLibrary
            self.present(pc, animated: true, completion: nil)
        }
    }

    // Resume real-time mode
    @IBAction func btnResumeRealTimeAction(_ sender: UIButton) {
        // Hide controls
        viewStillImage.isHidden = true

        // Clear results
        annotations = nil
        tableView.reloadData()

        // Resume real-time capture
        realTimeCaptureEnabled = true
        realTimeCapture(on: true)
    }

    // Turn capture on / off
    func realTimeCapture(on: Bool) {
        if cameraAvailable && realTimeCaptureEnabled && on  {
            // Start video capture
            vcc?.turnVideoCapture(on: true)

            // Setup timer
            self.scheduleCapture()
        } else {
            sampleTimer?.invalidate()
            vcc?.turnVideoCapture(on: false)
        }
    }

    // Schedule capture in next X seconds
    func scheduleCapture() {
        sampleTimer = Timer.scheduledTimer(withTimeInterval: 1.2, repeats: false, block: { timer in
            // Have vcc schedule next frame to be captured
            self.vcc?.captureNextFrame()
        })
    }

    // MARK: VideoFrameCaptureDelegate

    // Frame captured as UIImage
    func frameCapture(didCapture image: UIImage) {
        // Send request
        BPAIBackend.annotateImage(image: image) { (annotations, error) in
            if let e = error {
                self.alert(title: "Error", message: e)
                return
            }

            // Present results
            self.annotations = annotations
            self.tableView.reloadSections(IndexSet(integer: 0), with: UITableViewRowAnimation.fade)

            // Schedule next capture
            if self.realTimeCaptureEnabled {
                self.scheduleCapture()
            }
        }
    }

    // MARK: UITableView Delegates
    func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
        return 60
    }

    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        if let a = annotations {
            return a.count
        }
        return 0
    }

    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "AnnotationTableViewCell", for: indexPath) as! AnnotationTableViewCell

        // Data
        if let data = annotations {
            let annotation = data[indexPath.row]
            cell.lblAnnotationName.text = annotation.class_name
            cell.lblProbability.text = String(format: "%.3f", annotation.probability)
            cell.pbProbability.progress = annotation.probability
        } else {
            cell.lblAnnotationName.text = "No Data"
            cell.lblProbability.text = "0"
            cell.pbProbability.progress = 0
        }
        return cell
    }

    // MARK: UIImagePickerControllerDelegate
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        picker.dismiss(animated: true, completion: nil)

        // Picked still image
        if let image = info[UIImagePickerControllerOriginalImage] as? UIImage {
            // Stop real-time capture
            realTimeCaptureEnabled = false
            realTimeCapture(on: false)

            // Show still image controls
            viewStillImage.isHidden = false

            // Scale and crop image to 1:1
            let image = image.af_imageAspectScaled(toFill: CGSize(width: 1024, height: 1024))

            // Display image
            imgStillImage.image = image

            // Pass frame for annotation
            frameCapture(didCapture: image)
        }
    }
}
