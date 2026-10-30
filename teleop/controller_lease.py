from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ControllerLease:
    session_id: str | None = None
    last_seq: int = -1
    # Per-hand sequence tracking for bimanual mode.
    _hand_seqs: dict[str, int] = field(default_factory=dict)

    def claim(self, session_id: str) -> bool:
        if self.session_id not in {None, session_id}:
            return False
        if self.session_id != session_id:
            self.last_seq = -1
            self._hand_seqs.clear()
        self.session_id = session_id
        return True

    def release(self, session_id: str) -> bool:
        if self.session_id != session_id:
            return False
        self.session_id = None
        self.last_seq = -1
        self._hand_seqs.clear()
        return True

    def owns(self, session_id: str) -> bool:
        return self.session_id == session_id

    @property
    def last_seq_by_hand(self) -> dict[str, int]:
        return dict(self._hand_seqs)

    def accept_sequence(self, session_id: str, seq: int, hand: str | None = None) -> bool:
        if not self.owns(session_id):
            return False
        if hand is not None:
            hand_seq = self._hand_seqs.get(hand, -1)
            if seq <= hand_seq:
                return False
            self._hand_seqs[hand] = seq
            self.last_seq = max(self.last_seq, seq)
            return True
        # Legacy global sequence
        if seq <= self.last_seq:
            return False
        self.last_seq = seq
        return True
