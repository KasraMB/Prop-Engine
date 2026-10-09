from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class ProcessingCalendar:
    timezone: str = "America/New_York"
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)
    holidays: tuple[date, ...] = ()
    opening: time = time(9)
    closing: time = time(17)

    def __post_init__(self):
        ZoneInfo(self.timezone)
        object.__setattr__(self, "weekdays", tuple(self.weekdays))
        object.__setattr__(self, "holidays", tuple(self.holidays))
        if not self.weekdays or any(type(d) is not int or not 0 <= d <= 6 for d in self.weekdays):
            raise ValueError("processing weekdays must be numbers 0..6")
        if any(type(d) is not date for d in self.holidays):
            raise ValueError("processing holidays must be dates")
        if any(not isinstance(t, time) or t.tzinfo is not None for t in (self.opening, self.closing)):
            raise ValueError("processing hours must be local times")
        if self.opening >= self.closing:
            raise ValueError("processing opening must precede closing")

    def after(self, at, delay):
        if not isinstance(at, datetime) or at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("processing timestamp must be aware")
        if not isinstance(delay, timedelta) or delay < timedelta(0):
            raise ValueError("processing delay must be nonnegative")
        zone = ZoneInfo(self.timezone)
        local = (at.astimezone(timezone.utc) + delay).astimezone(zone)
        day = local.date()
        while True:
            if day.weekday() in self.weekdays and day not in self.holidays:
                opening = datetime.combine(day, self.opening, zone)
                closing = datetime.combine(day, self.closing, zone)
                candidate = max(local, opening)
                if candidate < closing:
                    return candidate.astimezone(timezone.utc)
            day += timedelta(days=1)
