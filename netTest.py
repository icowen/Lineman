import unittest
from unittest import mock

import numpy as np

import net


class DataTransformation(unittest.TestCase):
    def setUp(self):
        self.x, self.y, self.last_5_df = net.get_data()

    @mock.patch('net.train_model')
    @mock.patch('net.plot_loss')
    @mock.patch('net.plot_predictions')
    def test_data_not_nan(self, mock_train_model, mock_plot_loss, mock_plot_predictions):
        self.assertEqual(np.count_nonzero(np.isnan(self.x)), 0)
        self.assertEqual(np.count_nonzero(np.isnan(self.y)), 0)
        self.assertEqual(np.count_nonzero(self.last_5_df.isna()), 0)

    @mock.patch('net.train_model')
    @mock.patch('net.plot_loss')
    @mock.patch('net.plot_predictions')
    def test_data_shape(self, mock_train_model, mock_plot_loss, mock_plot_predictions):
        number_of_frames = 10188
        number_of_variables = 45
        x_shape = (number_of_frames, number_of_variables)
        y_shape = (number_of_frames,)
        self.assertEqual(self.x.shape, x_shape)
        self.assertEqual(self.y.shape, y_shape)
        self.assertEqual(self.last_5_df.shape, (235, 48))


if __name__ == '__main__':
    unittest.main()
