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

// BPAIBackend communicates with machine learning APIs deployed on blupig.net servers
import UIKit
import Alamofire
import AlamofireImage
import AlamofireObjectMapper
import ObjectMapper

class BPAIBackend: NSObject {
    private static let apiBaseURL: String = "https://api.blupig.net/ai"

    static func annotateImage(image: UIImage, completionHandler: @escaping ([ImageAnnotation]?, String?) -> Void) {
        // Prepare request
        // Resize image
        let scaledImage = image.af_imageAspectScaled(toFill: CGSize(width: 64, height: 64))
        let imageData = UIImageJPEGRepresentation(scaledImage, 0.8)
        if imageData == nil {
            completionHandler(nil, "annotateImage: Could not encode image")
            return
        }

        // Send request
        Alamofire.upload(multipartFormData: { multipartFormData in
            multipartFormData.append(imageData!, withName: "image", fileName: "image", mimeType: "image/jpeg")
        }, to: apiBaseURL + "/images/annotate", encodingCompletion: { encodingResult in
            // Check request encoding result
            switch encodingResult {
            case .success(request: let upload, streamingFromDisk: _, streamFileURL: _):
                upload.validate(statusCode: 200..<300).responseObject(completionHandler: { (response: DataResponse<ImageAnnotationResponse>) in
                    // Check response status
                    switch response.result {
                    case .success:
                        if let resultArray = response.result.value?.result {
                            completionHandler(resultArray, nil)
                        } else {
                            completionHandler(nil, "Request failed: empty result")
                        }
                        break
                    case .failure(let error):
                        completionHandler(nil, "Request failed: " + error.localizedDescription)
                    }
                })
                break
            case .failure(let encodingError):
                completionHandler(nil, "Could not encode request: " + encodingError.localizedDescription)
                break
            }
        })
    }
}

class ImageAnnotationResponse: Mappable {
    public var result: [ImageAnnotation]?
    public var error: String?

    required init?(map: Map) {}

    func mapping(map: Map) {
        result <- map["results"]
        error  <- map["error"]
    }
}

class ImageAnnotation: Mappable {
    public var class_name: String!
    public var probability: Float!

    required init?(map: Map) {}

    func mapping(map: Map) {
        class_name  <- map["class_name"]
        probability <- map["probability"]
    }
}
