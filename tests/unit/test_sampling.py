import pytest

from cyberslm.training.sampling import deterministic_batch_indices


def test_deterministic_sampling_resume_matches_uninterrupted_sequence() -> None:
    full: list[int] = []
    cursor = 0
    for _ in range(7):
        indices, cursor = deterministic_batch_indices(
            dataset_size=5, batch_size=2, sample_cursor=cursor, seed=2026
        )
        full.extend(indices)

    first: list[int] = []
    cursor = 0
    for _ in range(3):
        indices, cursor = deterministic_batch_indices(
            dataset_size=5, batch_size=2, sample_cursor=cursor, seed=2026
        )
        first.extend(indices)
    saved_cursor = cursor
    resumed: list[int] = []
    for _ in range(4):
        indices, cursor = deterministic_batch_indices(
            dataset_size=5, batch_size=2, sample_cursor=cursor, seed=2026
        )
        resumed.extend(indices)

    assert saved_cursor == 6
    assert first + resumed == full
    assert set(full[:5]) == set(range(5))


@pytest.mark.parametrize(
    ("dataset_size", "batch_size", "cursor"),
    [(0, 1, 0), (2, 0, 0), (2, 1, -1)],
)
def test_deterministic_sampling_rejects_invalid_arguments(
    dataset_size: int, batch_size: int, cursor: int
) -> None:
    with pytest.raises(ValueError):
        deterministic_batch_indices(
            dataset_size=dataset_size,
            batch_size=batch_size,
            sample_cursor=cursor,
            seed=1,
        )
