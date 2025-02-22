# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""One example for evaluate saved policy."""

import os
import omnisafe

# Just fill your experiment's log directory in here.
# Such as: ~/omnisafe/examples/runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2023-03-07-20-25-48
LOG_DIR = './runs/DD-{SafetyDrawCircle-v0}-SafetyDrawCircle-v0_data_expert/seed-000-2024-02-01-18-57-02'
# LOG_DIR = "./runs/DD-{SafetyPointCircle1-v0}-SafetyPointCircle1-v0-mixed-beta0.5/seed-000-2024-02-04-14-06-34"
if __name__ == '__main__':
    evaluator = omnisafe.Evaluator(render_mode='human')
    scan_dir = os.scandir(os.path.join(LOG_DIR, 'torch_save'))
    for item in scan_dir:
        if item.is_file() and item.name.split('.')[-1] == 'pt':
            evaluator.load_saved(
                save_dir=LOG_DIR,
                model_name=item.name,
                camera_name='track',
                width=256,
                height=256,
            )
            evaluator.render(num_episodes=5)
            # evaluator.evaluate(num_episodes=2)
    scan_dir.close()
