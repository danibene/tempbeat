import numpy as np

from tempbeat.misc.misc_utils import argtop_k, get_func_kwargs, top_k


class TestGetFuncKwargs:
    """
    Test cases for the get_func_kwargs function.
    """

    @staticmethod
    def function_with_defaults(a: int, b: int = 2, c: int = 3) -> int:
        """
        Example function with default values.

        Parameters
        ----------
        a : int
            The first parameter.
        b : int, optional
            The second parameter with a default value of 2.
        c : int, optional
            The third parameter with a default value of 3.

        Returns
        -------
        int
            The sum of the input parameters.
        """
        return a + b + c

    def test_basic_functionality(self) -> None:
        """
        Test basic functionality of get_func_kwargs.

        The function should correctly extract keyword arguments from function_with_defaults.
        """
        kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
        result = get_func_kwargs(self.function_with_defaults, **kwargs)
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_exclude_keys(self) -> None:
        """
        Test excluding keys from get_func_kwargs.

        The function should exclude specified keys from the extracted keyword arguments.
        """
        kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
        exclude_keys = ["c", "d"]
        result = get_func_kwargs(
            self.function_with_defaults, exclude_keys=exclude_keys, **kwargs
        )
        assert result == {"a": 1, "b": 2}

    def test_function_with_defaults(self) -> None:
        """
        Test get_func_kwargs with a function having default values.

        The function should correctly handle a function with default parameter values.
        """
        kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
        result = get_func_kwargs(self.function_with_defaults, **kwargs)
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_function_with_defaults_and_exclude_keys(self) -> None:
        """
        Test get_func_kwargs with a function having default values and exclude_keys.

        The function should correctly handle excluding keys from a function with default values.
        """
        kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
        exclude_keys = ["c"]
        result = get_func_kwargs(
            self.function_with_defaults, exclude_keys=exclude_keys, **kwargs
        )
        assert result == {"a": 1, "b": 2}


class TestArgTopK:
    @staticmethod
    def test_argtop_k_with_array() -> None:
        """
        Test argtop_k with a numpy array.
        """
        a = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
        result = argtop_k(a, k=2)
        expected = np.array([5, 7])
        np.testing.assert_array_equal(result, expected)

    @staticmethod
    def test_argtop_k_with_list() -> None:
        """
        Test argtop_k with a Python list.
        """
        a = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        result = argtop_k(a, k=2)
        expected = np.array([5, 7])
        np.testing.assert_array_equal(result, expected)

    @staticmethod
    def test_argtop_k_k_greater_than_length() -> None:
        """
        Test argtop_k with k greater than the length of the array.
        """
        a = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        result = argtop_k(a, k=20)
        expected = np.array([5, 7, 10, 8, 4, 2, 9, 0, 6, 3, 1])
        assert len(result) == len(expected)

    @staticmethod
    def test_argtop_k_empty_array() -> None:
        """
        Test argtop_k with an empty array.
        """
        a = np.array([])
        result = argtop_k(a, k=3)
        expected = np.array([])
        np.testing.assert_array_equal(result, expected)


class TestTopK:
    """
    Test cases for the top_k function.
    """

    @staticmethod
    def test_top_k_with_array() -> None:
        """
        Test top_k with a numpy array.

        The function should correctly return the top k elements from the input array.
        """
        a = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
        result = top_k(a, k=3)
        expected = np.array([9, 6, 5])
        np.testing.assert_array_equal(result, expected)

    @staticmethod
    def test_top_k_with_list() -> None:
        """
        Test top_k with a Python list.

        The function should correctly return the top k elements from the input list.
        """
        a = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        result = top_k(a, k=3)
        expected = [9, 6, 5]
        np.testing.assert_array_equal(result, expected)

    @staticmethod
    def test_top_k_k_greater_than_length() -> None:
        """
        Test top_k with k greater than the length of the array.

        The function should return all elements if k is greater than the length.
        """
        a = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        result = top_k(a, k=20)
        expected = [9, 6, 5, 5, 5, 4, 3, 3, 2, 1, 1]
        np.testing.assert_array_equal(result, expected)

    @staticmethod
    def test_top_k_empty_array() -> None:
        """
        Test top_k with an empty array.

        The function should return an empty array.
        """
        a = np.array([])
        result = top_k(a, k=3)
        expected = np.array([])
        np.testing.assert_array_equal(result, expected)
