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

import UIKit

class InfoTableViewController: UITableViewController {

    @IBOutlet weak var lblModelInfo: UILabel!

    override func viewDidLoad() {
        super.viewDidLoad()

        // Fetch API info
        BPAIBackend.apiInfo { (apiInfoResponse, error) in
            if error != nil {
                self.lblModelInfo.text = error
            } else {
                let loaded_model = apiInfoResponse?.loaded_model
                self.lblModelInfo.text = loaded_model
            }
        }
    }

    override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        tableView.deselectRow(at: indexPath, animated: true)
        if indexPath.section == 1 && indexPath.row == 0 {
            guard let url = URL(string: "https://github.com/yunzhu-li/recognizer") else {
                return
            }
            UIApplication.shared.open(url, options: [:], completionHandler: nil)
        }
    }
}
