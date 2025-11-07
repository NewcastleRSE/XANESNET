"""
XANESNET

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either Version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from torch import nn


class Model(nn.Module):
    """
    Abstract base class for XANESNET models.
    """

    def __init__(self):
        super().__init__()

        self.nn_flag = 0
        self.ae_flag = 0
        self.aegan_flag = 0
        self.gnn_flag = 0

        # 0 if forward() accepts a tensor x, 1 if forward() accepts a batch object
        self.batch_flag = 0
        self.config = {}

    def register_config(self, args, **kwargs):
        """
        Assign arguments from the child class's constructor to self.config.

        Args:
            args: The dictionary of arguments from the child class's constructor
            **kwargs: additional arguments to store
        """
        config = kwargs.copy()

        # Extract parameters from the local_vars, excluding 'self' and '__class__'
        args_dict = {
            key: val for key, val in args.items() if key not in ["self", "__class__"]
        }

        config.update(args_dict)

        self.config = config
