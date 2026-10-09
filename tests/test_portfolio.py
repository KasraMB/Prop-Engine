from datetime import datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction as F
from random import Random

import pytest

from propfirm_engine.events import Fill, Marks, merge_events, ordered
from propfirm_engine.instruments import Instrument
from propfirm_engine.portfolio import Book


AT = datetime(2026, 1, 5, 15, tzinfo=timezone.utc)
ES = Instrument("ES", 50, .25)
NQ = Instrument("NQ", 20, .25)


def fill(seq, quantity, price, fee=0, symbol="ES"):
    return Fill(AT, symbol, quantity, price, fee, seq)


def test_partial_exit_uses_fifo_and_preserves_open_profit():
    book = Book([ES], balance=50_000)
    for event in (fill(0, 2, 100, 1), fill(1, 1, 102, 1), fill(2, -1, 103, 1)):
        book.apply(event)
    state = book.snapshot()
    assert state.balance == 50_147
    assert state.realized == 150 and state.fees == 3
    assert state.unrealized == 200 and state.equity == 50_347
    position, = state.positions
    assert position.quantity == 2 and position.average_price == 101
    book.apply(fill(3, -2, 99, 2))
    assert book.balance == 49_945
    assert book.realized == -50 and book.fees == 5
    assert book.unrealized == 0 and not book.snapshot().positions


def test_reversal_opens_only_the_remaining_quantity():
    book = Book([ES])
    for event in (fill(0, 2, 100, 1), fill(1, -3, 102, 1)):
        book.apply(event)
    position, = book.snapshot().positions
    assert position.quantity == -1 and position.average_price == 102
    assert book.realized == 200 and book.balance == 198
    book.apply(fill(2, 1, 101, 1))
    assert book.realized == 250 and book.balance == 247
    assert book.equity == book.balance


def test_simultaneous_marks_value_the_portfolio_together():
    book = Book([ES, NQ], balance=50_000)
    book.apply(fill(0, 1, 100, 2))
    book.apply(fill(1, -2, 200, 3, "NQ"))
    book.apply(Marks(AT, (("ES", 101), ("NQ", 201)), seq=2))
    assert book.balance == 49_995
    assert book.unrealized == 10 and book.equity == 50_005
    assert {p.symbol: p.unrealized for p in book.snapshot().positions} == {"ES": 50, "NQ": -40}


def test_marks_and_fills_are_distinct_when_requested():
    book = Book([ES], mark_fills=False)
    with pytest.raises(ValueError, match="mark is required"):
        book.apply(fill(0, 1, 100.25))
    book.apply(Marks(AT, (("ES", 100),), seq=0))
    book.apply(fill(1, 1, 100.25, .5))
    assert book.balance == F("-.5") and book.unrealized == F("-12.5")
    book.apply(fill(2, -1, 99.75, .5))
    assert book.balance == -26 and book.unrealized == 0


def test_snapshot_retains_mark_age_and_does_not_change():
    book = Book([ES], mark_fills=False)
    assert book.snapshot().at is None
    book.apply(Marks(AT, (("ES", 100),)))
    later = AT + timedelta(minutes=5)
    book.apply(Fill(later, "ES", 1, 100))
    before = book.snapshot()
    assert before.at == later and before.positions[0].mark_at == AT
    book.apply(Marks(later, (("ES", 101),), seq=1))
    assert before.equity == 0 and book.equity == 50
    assert book.snapshot().positions[0].mark_at == later


def test_public_exports_preserve_instrument_import():
    from propfirm_engine import Book as PublicBook, Fill as PublicFill, Marks as PublicMarks
    from propfirm_engine.market_replay import Instrument as PriceInstrument
    assert PublicBook is Book and PublicFill is Fill and PublicMarks is Marks
    assert PriceInstrument is Instrument


def test_invalid_mark_batch_leaves_all_state_unchanged():
    book = Book([ES, NQ])
    book.apply(fill(0, 1, 100))
    before = book.snapshot()
    with pytest.raises(ValueError, match="tick grid"):
        book.apply(Marks(AT, (("ES", 105), ("NQ", 100.1)), seq=1))
    assert book.snapshot() == before
    book.apply(Marks(AT, (("ES", 105), ("NQ", 100)), seq=1))
    assert book.equity == 250


def test_unknown_instrument_and_invalid_fill_do_not_mutate_state():
    book = Book([ES])
    for event in (fill(0, 1, 100, symbol="NQ"), fill(0, 1, 100.1)):
        with pytest.raises(ValueError):
            book.apply(event)
        assert book.balance == 0 and not book.snapshot().positions
    book.apply(fill(0, 1, 100))


def test_decimal_prices_and_fractional_fees_are_exact():
    book = Book([Instrument("FX", 125_000, .00005)])
    book.apply(Fill(AT, "FX", 1, Decimal("1.10000"), F(1, 3)))
    book.apply(Fill(AT, "FX", -1, Decimal("1.10005"), F(1, 3), seq=1))
    assert book.realized == F("6.25")
    assert book.balance == F("6.25") - F(2, 3)


def test_negative_futures_prices_are_valid():
    book = Book([Instrument("CL", 1000, .01)])
    book.apply(Fill(AT, "CL", 1, -10))
    book.apply(Fill(AT, "CL", -1, -9, seq=1))
    assert book.balance == 1000


def test_signed_symmetry():
    books = [Book([ES]), Book([ES])]
    for seq, (quantity, price) in enumerate(((2, 100), (1, 102), (-1, 103), (-3, 99), (1, 98))):
        books[0].apply(fill(seq, quantity, price, .75))
        books[1].apply(fill(seq, -quantity, 200-price, .75))
        assert books[0].equity == books[1].equity
        assert books[0].realized == books[1].realized


def test_random_cash_reconciles_without_reusing_lot_accounting():
    rng = Random(42)
    book = Book([ES])
    quantity, mark, equity = 0, 400, F(0)
    for seq in range(2000):
        price = mark + rng.randint(-8, 8)
        delta = rng.choice((-3, -2, -1, 1, 2, 3))
        fee = F(7, 4) * abs(delta)
        equity += quantity * (price-mark) * F("12.5") - fee
        quantity += delta
        book.apply(fill(seq, delta, F(price, 4), fee))
        assert book.equity == equity
        assert book.balance + book.unrealized == equity
        mark = price


def test_closed_lots_are_not_retained():
    book = Book([ES])
    for seq in range(1000):
        book.apply(fill(seq, 1 if seq % 2 == 0 else -1, 100))
    assert not book._positions["ES"].lots
    assert not book.snapshot().positions


def test_fills_at_the_same_price_share_one_lot():
    book = Book([ES])
    for seq in range(100):
        book.apply(fill(seq, 1, 100))
    assert len(book._positions["ES"].lots) == 1
    assert book.snapshot().positions[0].quantity == 100


def test_event_order_is_explicit_and_checked_before_mutation():
    book = Book([ES])
    book.apply(fill(1, 1, 100))
    before = book.snapshot()
    for seq in (0, 1):
        with pytest.raises(ValueError, match="strictly increase"):
            book.apply(fill(seq, -1, 200))
        assert book.snapshot() == before


def test_merge_streams_and_reject_ambiguous_ties():
    first = [fill(0, 1, 100), fill(2, -1, 101)]
    second = [Marks(AT, (("ES", 100.5),), seq=1)]
    assert [e.seq for e in merge_events(first, second)] == [0, 1, 2]
    with pytest.raises(ValueError, match="strictly increase"):
        list(merge_events(first, [fill(0, 1, 100, symbol="NQ")]))
    with pytest.raises(ValueError, match="strictly increase"):
        list(ordered(reversed(first)))


def test_merge_is_lazy_and_normalizes_timezones():
    seen = []
    def feed():
        for i in range(100):
            seen.append(i)
            yield fill(i, 1, 100)
    stream = merge_events(feed())
    assert not seen
    assert next(stream).seq == 0 and seen == [0]
    local = AT.astimezone(timezone(timedelta(hours=-5)))
    assert Fill(local, "ES", 1, 100).at == AT


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "100", None])
def test_invalid_prices(value):
    with pytest.raises(ValueError):
        fill(0, 1, value)


@pytest.mark.parametrize("quantity", [0, 1.5, True, "1"])
def test_invalid_quantities(quantity):
    with pytest.raises(ValueError, match="quantity"):
        fill(0, quantity, 100)


@pytest.mark.parametrize("seq", [-1, .5, True])
def test_invalid_sequences(seq):
    with pytest.raises(ValueError, match="sequence"):
        fill(seq, 1, 100)


def test_invalid_contracts_and_event_fields():
    for instruments in ([], [ES, ES], ["ES"]):
        with pytest.raises(ValueError):
            Book(instruments)
    with pytest.raises(ValueError, match="timezone"):
        Fill(AT.replace(tzinfo=None), "ES", 1, 100)
    with pytest.raises(ValueError, match="fee"):
        fill(0, 1, 100, -1)
    for prices in ((), (("ES", 100), ("ES", 101))):
        with pytest.raises(ValueError):
            Marks(AT, prices)
    with pytest.raises(TypeError):
        list(ordered([object()]))
