# add jefapato from the upper directory to the path

import numpy as np
import pytest
from pytest import approx

from jefapato.facial_features import features 

def test_ear_score_inputs():
    # Test case 1: Invalid input types
    input_data = [1, 2, 3, 4, 5]
    with pytest.raises(TypeError):
        features.ear_score(input_data)
    
    # Test case 2: Invalid input types
    input_data = "invalid"
    with pytest.raises(TypeError):
        features.ear_score(input_data)

    # Test correct exception raises
    # Test case 3: Empty input
    input_data = None
    with pytest.raises(ValueError):
        features.ear_score(input_data)

    # Test case 4: Invalid input range
    input_data = np.array([1, 2, 3, 4, 6])
    with pytest.raises(ValueError):
        features.ear_score(input_data)
    
    # TEST DEPRIECATED
    # Reason: negative values are possible (but rare) and should not raise an error
    # # Test case 5: Invalid input range
    # input_data = np.array([[-2, 2], [1,4], [4,4], [5,2], [4, 0], [1, 0]])
    # with pytest.raises(ValueError):
    #     features.ear_score(input_data)
        
    # Test case 6: Out of range, thus not valid output
    input_data = np.array([[0, 2], [1,10], [4,10], [5,2], [4, 0], [1, 0]])
    score, valid = features.ear_score(input_data)
    assert not valid
 
def test_ear_score():
    # Test case 1: Valid input of sphere
    input_data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    expected_output = 1.0
    assert features.ear_score(input_data)[0] == approx(expected_output)

    # Test case 2: Valid input
    input_data = np.array([[0, 2], [1,4], [4,4], [5,2], [4, 0], [1, 0]])
    expected_output = (4 + 4) / 10
    assert features.ear_score(input_data)[0] == approx(expected_output)

    # Test case 3: Valid input
    input_data = np.array([[0, 2], [1,3], [4,3], [5,2], [4, 1], [1, 1]])
    expected_output = (2 + 2) / 10
    assert features.ear_score(input_data)[0] == approx(expected_output)
