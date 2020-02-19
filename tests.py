import pytest
import numpy as np

from iou import iou


TEST_DATA = [
    # rect 1, rect 2, iou, comment
    ([0, 4, 10, 10], [0, 5, 11, 10], 0.7692307692307693, "small intersection"),
    ([12, 15, 32, 42], [4, 9, 30, 55], 0.3888, "big intersection"),
    ([11, 70, 256, 1000], [11, 70, 256, 1000], 1.0, "same rectangles"),
    ([99, 110, 111, 220], [101, 115, 102, 116], 0.0007575757575757576, "one include other"),
    ([0, 0, 100, 50], [110, 0, 150, 50], 0.0, "no intersection horizontal"),
    ([0, 0, 100, 50], [10, 64, 90, 96], 0.0, "no intersection vertical"),
    ([30, 40, 50, 60], [0, 10, 20, 30], 0.0, "no intersection diagonal"),
    ([45, 10, 60, 39], [60, 25, 75, 50], 0.0, "side connection"),
    ([45, 10, 60, 39], [60, 39, 66, 45], 0.0, "corner connection"),
]


# wrong points order


@pytest.mark.parametrize("rect1, rect2, exp_iou, comment", TEST_DATA)
def test_positive_cases(rect1, rect2, exp_iou, comment):
    rect_batch1 = np.array([rect1], dtype='uint')
    rect_batch2 = np.array([rect2], dtype='uint')
    exp_iou = np.array([exp_iou], dtype='float32')

    res_iou = iou(rect_batch1, rect_batch2)

    assert np.allclose(res_iou, exp_iou), comment


def test_positive_batch():
    rect_batch1, rect_batch2, exp_iou, _ = zip(*TEST_DATA)

    rect_batch1 = np.array(rect_batch1, dtype='uint')
    rect_batch2 = np.array(rect_batch2, dtype='uint')
    exp_iou = np.array(exp_iou, dtype='float32')

    res_iou = iou(rect_batch1, rect_batch2)

    assert np.allclose(res_iou, exp_iou)


def test_positive_float():
    rect_batch1, rect_batch2, exp_iou, _ = zip(*TEST_DATA)

    rect_batch1 = np.array(rect_batch1, dtype='float32')
    rect_batch2 = np.array(rect_batch2, dtype='float32')
    exp_iou = np.array(exp_iou, dtype='float32')

    res_iou = iou(rect_batch1, rect_batch2)

    assert np.allclose(res_iou, exp_iou)


def test_mix_int_and_float():
    rect_batch1 = np.array([[0, 4, 10, 10]], dtype='uint')
    rect_batch2 = np.array([[0, 5, 11, 10]], dtype='float32')

    res_iou = iou(rect_batch1, rect_batch2)

    assert np.allclose(res_iou, np.array([0.7692307692307693], dtype='float32'))


def test_zero_area_rect():
    res_iou = iou(np.array([[10, 10, 10, 20]]), np.array([[0, 0, 100, 100]]))
    assert res_iou[0] == 0.0

    res_iou = iou(np.array([[5, 42, 10, 42]]), np.array([[5, 42, 10, 42]]))
    assert np.isnan(res_iou[0])


def test_empty_batches():
    with pytest.raises(IndexError):
        iou(np.array([]), np.array([]))


def test_different_batch_size():
    with pytest.raises(ValueError):
        iou(np.array([[0, 0, 10, 20], [5, 5, 8, 8]]), np.array([[0, 0, 10, 40]]))


def test_rect_not_a_batch():
    with pytest.raises(IndexError):
        iou(np.array([0, 0, 10, 20]), np.array([0, 0, 10, 40]))


def test_wrong_rect_size():
    with pytest.raises(AssertionError):
        iou(np.array([[1, 2, 3, 4, 5]]), np.array([[0, 1, 5, 7, 9]]))
