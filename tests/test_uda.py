import numpy as np
import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import xarray as xr

from toksearch.signal.uda import UDASignal


class TestUDASignal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_init_default_parameters(self):
        """Test UDASignal initialization with default parameters"""
        signal = UDASignal(treepath="test/signal")
        self.assertEqual(signal.treepath, "test/signal")
        self.assertEqual(signal.dims, ("times",))
        self.assertTrue(signal.fetch_units)
        self.assertFalse(signal.is_image)
        self.assertEqual(signal.timeout, 10)

    def test_init_custom_parameters(self):
        """Test UDASignal initialization with custom parameters"""
        signal = UDASignal(
            treepath="custom/signal",
            dims=("times", "channels"),
            fetch_units=False,
            is_image=True,
            timeout=30,
        )
        self.assertEqual(signal.treepath, "custom/signal")
        self.assertEqual(signal.dims, ("times", "channels"))
        self.assertFalse(signal.fetch_units)
        self.assertTrue(signal.is_image)
        self.assertEqual(signal.timeout, 30)

    @patch("pyuda.Client")
    def test_gather_basic_signal(self, MockClient):
        """Test gathering a basic 1D signal"""
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        # Create mock signal
        mock_signal = MagicMock()
        mock_dim = MagicMock()
        mock_dim.label = "time"
        mock_dim.data = np.linspace(0, 1, 100)
        mock_dim.units = "s"
        mock_signal.dims = [mock_dim]
        mock_signal.data = np.random.random(100)
        mock_signal.units = "V"

        mock_client.get.return_value = mock_signal

        signal = UDASignal(treepath="test/signal")
        result = signal.gather(12345)

        # Check that client was called correctly
        mock_client.set_property.assert_any_call("get_meta", True)
        mock_client.set_property.assert_any_call("timeout", 10)
        mock_client.get.assert_called_once_with("test/signal", 12345)

        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn("data", result)
        self.assertIn("times", result)
        self.assertIn("units", result)
        self.assertIsInstance(result["data"], np.ndarray)
        self.assertIsInstance(result["times"], np.ndarray)
        self.assertEqual(result["data"].shape, (100,))
        self.assertEqual(result["times"].shape, (100,))

    @patch("pyuda.Client")
    def test_gather_without_units(self, MockClient):
        """Test gathering signal without units"""
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        mock_signal = MagicMock()
        mock_dim = MagicMock()
        mock_dim.label = "time"
        mock_dim.data = np.linspace(0, 1, 100)
        mock_dim.units = "s"
        mock_signal.dims = [mock_dim]
        mock_signal.data = np.random.random(100)
        mock_signal.units = "V"

        mock_client.get.return_value = mock_signal

        signal = UDASignal(treepath="test/signal", fetch_units=False)
        result = signal.gather(12345)

        self.assertNotIn("units", result)

    @patch("pyuda.Client")
    def test_gather_multidimensional_signal(self, MockClient):
        """Test gathering a 2D signal"""
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        # Create mock 2D signal
        mock_dim1 = MagicMock()
        mock_dim1.label = "time"
        mock_dim1.data = np.linspace(0, 1, 50)
        mock_dim1.units = "s"

        mock_dim2 = MagicMock()
        mock_dim2.label = "channel"
        mock_dim2.data = np.arange(10)
        mock_dim2.units = ""

        mock_signal = MagicMock()
        mock_signal.dims = [mock_dim1, mock_dim2]
        mock_signal.data = np.random.random((50, 10))
        mock_signal.units = "V"

        mock_client.get.return_value = mock_signal

        signal = UDASignal(treepath="test/signal", dims=("times", "channels"))
        result = signal.gather(12345)

        self.assertIn("data", result)
        self.assertIn("times", result)
        self.assertIn("channels", result)
        self.assertEqual(result["data"].shape, (50, 10))
        self.assertEqual(result["times"].shape, (50,))
        self.assertEqual(result["channels"].shape, (10,))

    @patch("pyuda.Client")
    def test_gather_image_signal(self, MockClient):
        """Test gathering an image signal"""
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        # Create mock image
        mock_frame = MagicMock()
        mock_frame.k = np.random.random((480, 640))

        mock_image = MagicMock()
        mock_image.frames = [mock_frame, mock_frame, mock_frame]
        mock_image.frame_times = np.array([0.1, 0.2, 0.3])
        mock_image.is_color = False
        mock_image.height = 480
        mock_image.width = 640

        mock_client.get_images.return_value = mock_image

        signal = UDASignal(
            treepath="test/image", is_image=True, dims=("time", "height", "width")
        )
        result = signal.gather(12345)

        mock_client.get_images.assert_called_once_with("test/image", 12345)
        self.assertIn("data", result)
        self.assertIn("time", result)
        self.assertIn("height", result)
        self.assertIn("width", result)

    @patch("pyuda.Client")
    def test_gather_server_exception(self, MockClient):
        """Test handling of server exceptions"""
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        import pyuda

        mock_client.get.side_effect = pyuda.ServerException("Test error")

        signal = UDASignal(treepath="test/signal")

        with self.assertRaises(RuntimeError) as context:
            signal.gather(12345)

        self.assertIn("Could not load signal", str(context.exception))
        self.assertIn("test/signal", str(context.exception))

    @patch("pyuda.Client")
    def test_load_signal(self, MockClient):
        """Test load_signal method"""
        mock_client = MagicMock()

        mock_dim = MagicMock()
        mock_dim.label = "time"
        mock_dim.data = np.linspace(0, 1, 100)
        mock_dim.units = "s"

        mock_signal = MagicMock()
        mock_signal.dims = [mock_dim]
        mock_signal.data = np.random.random(100)
        mock_signal.units = "V"

        mock_client.get.return_value = mock_signal

        signal = UDASignal(treepath="test/signal")
        result = signal.load_signal(mock_client, 12345, "test/signal")

        self.assertIsInstance(result, xr.DataArray)
        self.assertEqual(result.name, "test/signal")

    def test_normalize_dimension_names(self):
        """Test dimension name normalization"""
        signal = UDASignal(treepath="test/signal")

        mock_dim1 = MagicMock()
        mock_dim1.label = "Time"

        mock_dim2 = MagicMock()
        mock_dim2.label = ""

        mock_dim3 = MagicMock()
        mock_dim3.label = "Channel-1"

        mock_signal = MagicMock()
        mock_signal.dims = [mock_dim1, mock_dim2, mock_dim3]

        result = signal._normalize_dimension_names(mock_signal)

        self.assertEqual(result[0], "time")
        self.assertEqual(result[1], "dim_0")
        self.assertEqual(result[2], "channel1")

    def test_normalize_dimension_names_special_chars(self):
        """Test dimension name normalization with special characters"""
        signal = UDASignal(treepath="test/signal")

        mock_dim = MagicMock()
        mock_dim.label = "Test@Dim#Name$"

        mock_signal = MagicMock()
        mock_signal.dims = [mock_dim]

        result = signal._normalize_dimension_names(mock_signal)

        self.assertEqual(result[0], "testdimname")

    def test_get_dataset_attributes(self):
        """Test extraction of dataset attributes"""
        signal = UDASignal(treepath="test/signal")

        mock_signal = MagicMock()
        mock_signal.units = "V"
        mock_signal.label = "Test Signal"
        mock_signal.description = "A test signal"
        mock_signal.rank = 1
        mock_signal.shape = (100,)
        mock_signal.data = np.array([1, 2, 3])
        result = signal._get_dataset_attributes("test", mock_signal)

        self.assertIn("units", result)
        self.assertIn("label", result)
        self.assertIn("description", result)
        self.assertNotIn("rank", result)
        self.assertNotIn("shape", result)
        self.assertNotIn("data", result)

    def test_get_dataset_attributes_numpy_types(self):
        """Test that numpy types are converted to Python types"""
        signal = UDASignal(treepath="test/signal")

        mock_signal = MagicMock()
        mock_signal.scalar_value = np.float64(3.14)
        mock_signal.array_value = np.array([1, 2, 3])
        mock_signal.tuple_value = (1, 2, 3)
        mock_signal.none_value = None
        mock_signal.data = np.array([])
        # Configure __dir__ to return only the attributes we set
        mock_signal.__dir__ = MagicMock(
            return_value=[
                "scalar_value",
                "array_value",
                "tuple_value",
                "none_value",
                "data",
            ]
        )

        result = signal._get_dataset_attributes("test", mock_signal)

        if "scalar_value" in result:
            self.assertIsInstance(result["scalar_value"], (int, float))
        if "array_value" in result:
            self.assertIsInstance(result["array_value"], list)
        if "tuple_value" in result:
            self.assertIsInstance(result["tuple_value"], list)
        if "none_value" in result:
            self.assertEqual(result["none_value"], "null")

    def test_cleanup_shot(self):
        """Test cleanup_shot method (should be a no-op)"""
        signal = UDASignal(treepath="test/signal")
        # Should not raise any exception
        signal.cleanup_shot(12345)

    def test_cleanup(self):
        """Test cleanup method (should be a no-op)"""
        signal = UDASignal(treepath="test/signal")
        # Should not raise any exception
        signal.cleanup()

    @patch("pyuda.Client")
    def test_fetch(self, MockClient):
        """Test the fetch method (inherited from Signal class)"""
        mock_client_instance = MagicMock()
        MockClient.return_value = mock_client_instance

        # Mock methods on the client if needed
        mock_dim = MagicMock()
        mock_dim.label = "time"
        mock_dim.data = np.linspace(0, 1, 1000)
        mock_dim.units = "s"

        mock_signal = MagicMock()
        mock_signal.dims = [mock_dim]
        mock_signal.data = np.random.random(1000)
        mock_signal.units = "V"

        mock_client_instance.get.return_value = mock_signal

        ip_signal = UDASignal(treepath="ip")
        result = ip_signal.fetch(30421)

        self.assertIsInstance(result, dict)
        self.assertIn("data", result)
        self.assertIsInstance(result["data"], np.ndarray)
        self.assertEqual(result["data"].shape, (1000,))
        self.assertIsInstance(result["times"], np.ndarray)
        self.assertEqual(result["times"].shape, (1000,))

    @patch("pyuda.Client")
    def test_convert_signal_to_dataset(self, MockClient):
        """Test _convert_signal_to_dataset method"""
        signal = UDASignal(treepath="test/signal")

        mock_dim = MagicMock()
        mock_dim.label = "time"
        mock_dim.data = np.linspace(0, 1, 100)
        mock_dim.units = "s"

        mock_signal = MagicMock()
        mock_signal.dims = [mock_dim]
        mock_signal.data = np.random.random(100)
        mock_signal.units = "V"

        result = signal._convert_signal_to_dataset("test_signal", mock_signal)

        self.assertIsInstance(result, xr.DataArray)
        self.assertEqual(result.name, "test_signal")
        self.assertEqual(result.attrs["name"], "test_signal")
        self.assertEqual(result.attrs["uda_name"], "test_signal")

    @patch("pyuda.Client")
    def test_convert_signal_named_time(self, MockClient):
        """Test that signals named 'time' are renamed to 'time_'"""
        signal = UDASignal(treepath="time")

        mock_dim = MagicMock()
        mock_dim.label = "time"
        mock_dim.data = np.linspace(0, 1, 100)
        mock_dim.units = "s"

        mock_signal = MagicMock()
        mock_signal.dims = [mock_dim]
        mock_signal.data = np.linspace(0, 1, 100)
        mock_signal.units = "s"

        result = signal._convert_signal_to_dataset("time", mock_signal)

        self.assertEqual(result.name, "time_")
        self.assertEqual(result.attrs["uda_name"], "time")

    def test_get_signal_metadata_fields(self):
        """Test retrieval of signal metadata fields"""
        signal = UDASignal(treepath="test/signal")

        mock_signal = MagicMock()
        mock_signal.units = "V"
        mock_signal.label = "Test"
        mock_signal.data = np.array([1, 2, 3])
        mock_signal.meta = {}
        mock_signal._private = "hidden"

        # Mock callable method
        mock_signal.some_method = MagicMock()

        # Configure __dir__ to return all attributes including private ones
        mock_signal.__dir__ = MagicMock(
            return_value=["units", "label", "data", "meta", "_private", "some_method"]
        )

        result = signal._get_signal_metadata_fields(mock_signal, "test")

        self.assertIn("units", result)
        self.assertIn("label", result)
        self.assertNotIn("data", result)
        self.assertNotIn("meta", result)
        self.assertNotIn("_private", result)
        self.assertNotIn("some_method", result)
