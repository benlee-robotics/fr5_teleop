from teleop.controller_lease import ControllerLease


def test_only_one_controller_owns_the_lease() -> None:
    lease = ControllerLease()
    assert lease.claim("one")
    assert not lease.claim("two")
    assert lease.accept_sequence("one", 1)
    assert not lease.accept_sequence("one", 1)
    assert lease.release("one")
    assert lease.claim("two")


def test_bimanual_sequences_are_independent_and_acknowledged() -> None:
    lease = ControllerLease()
    assert lease.claim("controller")

    assert lease.accept_sequence("controller", 0, "left")
    assert lease.accept_sequence("controller", 0, "right")
    assert not lease.accept_sequence("controller", 0, "left")
    assert lease.accept_sequence("controller", 1, "left")

    assert lease.last_seq == 1
    assert lease.last_seq_by_hand == {"left": 1, "right": 0}

    assert lease.release("controller")
    assert lease.last_seq == -1
    assert lease.last_seq_by_hand == {}
