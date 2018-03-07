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

class LicensesTableViewController: UITableViewController {

    // [[Name, URL]]
    private var licenses: [[String]] = []

    override func viewDidLoad() {
        super.viewDidLoad()

        licenses = [
            ["Tensorflow", "https://github.com/tensorflow/tensorflow/blob/master/LICENSE"],
            ["Keras", "https://github.com/keras-team/keras/blob/master/LICENSE"],
            ["numpy", "https://docs.scipy.org/doc/numpy-1.14.0/license.html"],
            ["scipy", "https://www.scipy.org/scipylib/license.html"],
            ["Pillow", "https://github.com/python-pillow/Pillow/blob/master/LICENSE"],
            ["h5py", "http://docs.h5py.org/en/latest/licenses.html"],
            ["Flask", "http://flask.pocoo.org/docs/0.12/license/"]
        ]

        tableView.reloadData()
    }

    // MARK: - Table view
    override func numberOfSections(in tableView: UITableView) -> Int {
        return 1
    }

    override func tableView(_ tableView: UITableView, titleForHeaderInSection section: Int) -> String? {
        return "Open-source Licenses"
    }

    override func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return licenses.count
    }

    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "LicenseTableViewCell", for: indexPath)

        let l = licenses[indexPath.row]
        cell.textLabel?.text = l[0]

        return cell
    }

    override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        tableView.deselectRow(at: indexPath, animated: true)

        let l = licenses[indexPath.row]
        guard let url = URL(string: l[1]) else {
            return
        }
        UIApplication.shared.open(url, options: [:], completionHandler: nil)
    }
}
