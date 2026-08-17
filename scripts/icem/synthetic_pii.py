"""Deterministic synthetic PII injection utilities."""

from __future__ import annotations

from dataclasses import dataclass
import random

from .schema import Span


PEOPLE = ("Alex Mercer", "Jordan Vale", "Mira Stone", "Samir Vale")
HANDLES = ("@river_user", "@north_test", "@sample_mira", "@civic_case")
EMAILS = ("mira.example@test.invalid", "alex.mercer@example.invalid")
PHONES = ("555-0104", "555-0188")
LOCATIONS = ("Northbridge", "Riverton", "Westford")
SCHOOLS = ("Riverton High", "Northbridge College")
WORKPLACES = ("CivicWorks Lab", "Westford Library")
DATES = ("2026-04-17", "May 3", "Friday at 8pm")
FAMILY_RELATIONS = ("cousin", "older brother", "aunt")
EVENTS = ("Riverton forum", "Northbridge meetup")


@dataclass(frozen=True)
class InjectedText:
    text: str
    spans: tuple[Span, ...]
    template: str


def _span_for(value: str, text: str, label: str) -> Span:
    start = text.index(value)
    return Span(start, start + len(value), label, "synthetic", replacement=f"[{label}]")


def inject_synthetic_pii(text: str, *, seed: int, row_index: int = 0) -> InjectedText:
    """Inject clearly fake PII and return gold spans for residual metrics."""

    rng = random.Random(seed + row_index)
    template = rng.choice(
        (
            "handle_prefix",
            "person_school_prefix",
            "email_suffix",
            "date_city_prefix",
            "family_prefix",
            "workplace_suffix",
        )
    )
    if template == "handle_prefix":
        handle = rng.choice(HANDLES)
        injected = f"{handle} said: {text}"
        return InjectedText(
            text=injected,
            spans=(Span(0, len(handle), "HANDLE", "synthetic", replacement="[HANDLE]"),),
            template=template,
        )
    if template == "person_school_prefix":
        person = rng.choice(PEOPLE)
        school = rng.choice(SCHOOLS)
        prefix = f"{person} from {school} wrote: "
        person_span = Span(0, len(person), "PERSON", "synthetic", replacement="[PERSON]")
        school_start = len(person) + len(" from ")
        school_span = Span(
            school_start,
            school_start + len(school),
            "SCHOOL",
            "synthetic",
            replacement="[SCHOOL]",
        )
        return InjectedText(
            text=f"{prefix}{text}",
            spans=(person_span, school_span),
            template=template,
        )
    if template == "date_city_prefix":
        date = rng.choice(DATES)
        person = rng.choice(PEOPLE)
        city = rng.choice(LOCATIONS)
        injected = f"On {date}, {person} in {city} posted: {text}"
        return InjectedText(
            text=injected,
            spans=(
                _span_for(date, injected, "DATE"),
                _span_for(person, injected, "PERSON"),
                _span_for(city, injected, "LOCATION"),
            ),
            template=template,
        )
    if template == "family_prefix":
        relation = rng.choice(FAMILY_RELATIONS)
        person = rng.choice(PEOPLE)
        phone = rng.choice(PHONES)
        injected = f"My {relation} {person} ({phone}) said {text}"
        return InjectedText(
            text=injected,
            spans=(
                _span_for(relation, injected, "FAMILY_RELATION"),
                _span_for(person, injected, "PERSON"),
                _span_for(phone, injected, "PHONE"),
            ),
            template=template,
        )
    if template == "workplace_suffix":
        workplace = rng.choice(WORKPLACES)
        event = rng.choice(EVENTS)
        injected = f"{text} - shared near {workplace} after the {event}"
        return InjectedText(
            text=injected,
            spans=(
                _span_for(workplace, injected, "WORKPLACE"),
                _span_for(event, injected, "EVENT"),
            ),
            template=template,
        )
    email = rng.choice(EMAILS)
    suffix = f" Contact: {email}"
    start = len(text) + len(" Contact: ")
    return InjectedText(
        text=f"{text}{suffix}",
        spans=(Span(start, start + len(email), "EMAIL", "synthetic", replacement="[EMAIL]"),),
        template=template,
    )
