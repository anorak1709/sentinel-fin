"""
Synthetic authentication + transaction log generator for Sentinel-Fin.

Produces two JSONL files:
  - data/logs/auth_events.jsonl
  - data/logs/transaction_events.jsonl

The data contains a baseline of normal user activity plus deliberately
injected attack chains (ATO with impossible travel, brute-force precursor,
new-device + high-value transfer) so the correlation engine has known
positives to detect.
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

from faker import Faker    #Faker is used for generating realistic IP addresses, user agents, etc. 

fake = Faker("en_IN") # "en_IN" locale gives us more India-specific data, which is our target demographic for this simulation.
random.seed(42) # Seeding the random number generator for reproducibility. This ensures that every run of this script produces the same synthetic dataset, which is important for consistent evaluation.
Faker.seed(42)


# ---------- Config ----------

NUM_USERS = 500
NUM_NORMAL_DAYS = 7
EVENTS_PER_USER_PER_DAY = (3, 12)   # uniform range
ATTACK_CHAINS_TO_INJECT = 25         # known positives for evaluation

#Total attacks will be 25000 out of which 25 will be ATO with impossible travel, brute-force precursor, new-device + high-value transfer. This allows us to test the correlation engine's ability to detect complex attack patterns while maintaining a realistic ratio of normal to malicious activity.

OUTPUT_DIR = Path("data/logs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Geo lookup — (city, country, lat, lon)
# Mix of Indian cities (normal) + foreign cities (suspicious)
INDIA_CITIES = [
    ("Bengaluru", "IN", 12.9716, 77.5946),
    ("Mumbai",    "IN", 19.0760, 72.8777),
    ("Delhi",     "IN", 28.6139, 77.2090),
    ("Chennai",   "IN", 13.0827, 80.2707),
    ("Hyderabad", "IN", 17.3850, 78.4867),
    ("Pune",      "IN", 18.5204, 73.8567),
    ("Kolkata",   "IN", 22.5726, 88.3639),
]

FOREIGN_CITIES = [
    ("Lagos",     "NG",  6.5244,  3.3792),
    ("Moscow",    "RU", 55.7558, 37.6173),
    ("Kyiv",      "UA", 50.4501, 30.5234),
    ("Bucharest", "RO", 44.4268, 26.1025),
    ("Manila",    "PH", 14.5995, 120.9842),
]

MERCHANT_CATEGORIES = [
    "grocery", "fuel", "utility", "telecom", "ecommerce",
    "travel", "restaurant", "p2p_transfer", "atm_withdrawal",
]


# ---------- Data classes ----------

@dataclass
class AuthEvent:
    event_id: str
    timestamp: str
    user_id: str
    event_type: str          # "login_success" | "login_failure"
    ip_address: str
    city: str
    country: str
    latitude: float
    longitude: float
    device_fingerprint: str
    user_agent: str

@dataclass
class TransactionEvent:
    event_id: str
    timestamp: str
    user_id: str
    amount_inr: float
    payee_id: str
    payee_is_new: bool
    channel: str             # "upi" | "neft" | "card" | "imps"
    merchant_category: str
    device_fingerprint: str


# ---------- User profile generation ----------

@dataclass
class UserProfile:
    user_id: str
    home_city: tuple
    primary_device: str
    known_payees: list
    typical_amount_range: tuple
    active_hours: tuple      # (start, end) in 24h

def make_user(i: int) -> UserProfile:
    home = random.choice(INDIA_CITIES)
    typical_low = random.choice([50, 100, 200, 500])
    typical_high = typical_low * random.choice([10, 20, 50])
    return UserProfile(
        user_id=f"U_{i:04d}",
        home_city=home,
        primary_device=f"dev_{uuid.uuid4().hex[:10]}",
        known_payees=[f"P_{uuid.uuid4().hex[:8]}" for _ in range(random.randint(3, 10))],
        typical_amount_range=(typical_low, typical_high),
        active_hours=(random.choice([6, 7, 8, 9]), random.choice([21, 22, 23])),
    )


# ---------- Normal activity generation ----------

def gen_normal_events(
    user: UserProfile, day_start: datetime
) -> Iterator[tuple[str, AuthEvent | TransactionEvent]]:
            #Yield ('auth', AuthEvent) or ('txn', TransactionEvent) tuples for one user on one day.
    n_events = random.randint(*EVENTS_PER_USER_PER_DAY)
    for _ in range(n_events):
        # Pick an hour inside the user's active window
        hour = random.randint(*user.active_hours)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        ts = day_start.replace(hour=hour, minute=minute, second=second)

        # 30% chance of a login event, 70% transaction
        if random.random() < 0.30:
            yield "auth", AuthEvent(
                event_id=str(uuid.uuid4()),
                timestamp=ts.isoformat(),
                user_id=user.user_id,
                event_type="login_success" if random.random() > 0.05 else "login_failure", #Even a user with normal behavior can have occasional failed logins due to mistyped passwords, network issues, etc. We set a 5% failure rate for realism.
                ip_address=fake.ipv4_public(),
                city=user.home_city[0],
                country=user.home_city[1],
                latitude=user.home_city[2] + random.uniform(-0.05, 0.05),  #Adding an offset as a user may not always login from the same place that he stays at. For eg, a working individual may login from a location near his office during working hours and from a location near his home during non-working hours.   
                longitude=user.home_city[3] + random.uniform(-0.05, 0.05),
                device_fingerprint=user.primary_device,
                user_agent=fake.user_agent(),
            )
        else:
            yield "txn", TransactionEvent(
                event_id=str(uuid.uuid4()),
                timestamp=ts.isoformat(),
                user_id=user.user_id,
                amount_inr=round(random.uniform(*user.typical_amount_range), 2),
                payee_id=random.choice(user.known_payees),
                payee_is_new=False,
                channel=random.choice(["upi", "upi", "upi", "card", "neft", "imps"]),
                merchant_category=random.choice(MERCHANT_CATEGORIES),
                device_fingerprint=user.primary_device,
            )


# ---------- Attack chain injection ----------

def gen_ato_attack_chain(
    user: UserProfile, attack_time: datetime
) -> list[tuple[str, AuthEvent | TransactionEvent]]:
    """
    Inject a realistic Account Takeover chain:
      1. Burst of 5-8 failed logins from a foreign IP (brute force / cred stuffing)
      2. One successful login from the same foreign IP using a NEW device
      3. A high-value transfer to a brand-new payee within 5 minutes of login
    """
    foreign = random.choice(FOREIGN_CITIES)
    attacker_ip = fake.ipv4_public()
    attacker_device = f"dev_{uuid.uuid4().hex[:10]}"
    new_payee = f"P_NEW_{uuid.uuid4().hex[:8]}"

    events: list[tuple[str, AuthEvent | TransactionEvent]] = []

    # Brute-force burst (MITRE T1110 - Brute Force: The time between failed attempts is short, indicating an automated attack trying many passwords in quick succession.)
    n_failures = random.randint(5, 8)
    for i in range(n_failures):
        ts = attack_time + timedelta(seconds=i * random.randint(8, 25))
        events.append(("auth", AuthEvent(
            event_id=str(uuid.uuid4()),
            timestamp=ts.isoformat(),
            user_id=user.user_id,
            event_type="login_failure",
            ip_address=attacker_ip,
            city=foreign[0],
            country=foreign[1],
            latitude=foreign[2],
            longitude=foreign[3],
            device_fingerprint=attacker_device,
            user_agent=fake.user_agent(),
        )))

    # Successful breach(MITRE T1078 - Valid Accounts: The attacker successfully logs in using valid credentials, which may have been obtained through the brute-force attack or from a previous data breach. The login is from a new device and a foreign location, which are strong indicators of compromise.)
    success_time = attack_time + timedelta(seconds=n_failures * 20 + 5)
    events.append(("auth", AuthEvent(
        event_id=str(uuid.uuid4()),
        timestamp=success_time.isoformat(),
        user_id=user.user_id,
        event_type="login_success",
        ip_address=attacker_ip,
        city=foreign[0],
        country=foreign[1],
        latitude=foreign[2],
        longitude=foreign[3],
        device_fingerprint=attacker_device,
        user_agent=fake.user_agent(),
    )))

    # High-value transfer to new payee (MITRE T1110 + T1078: The attacker, having gained access to the account, initiates a high-value transaction to a new payee. This action is consistent with the objectives of an account takeover attack, where the attacker seeks to quickly extract value from the compromised account. The transaction occurs shortly after the successful login, indicating that the attacker is acting swiftly to avoid detection.)
    txn_time = success_time + timedelta(seconds=random.randint(60, 240))
    high_amount = user.typical_amount_range[1] * random.uniform(3, 8)
    events.append(("txn", TransactionEvent(
        event_id=str(uuid.uuid4()),
        timestamp=txn_time.isoformat(),
        user_id=user.user_id,
        amount_inr=round(high_amount, 2),
        payee_id=new_payee,
        payee_is_new=True,
        channel=random.choice(["upi", "imps", "neft"]),
        merchant_category="p2p_transfer",
        device_fingerprint=attacker_device,
    )))

    return events


# ---------- Main pipeline ----------

def main() -> None:
    users = [make_user(i) for i in range(NUM_USERS)]
    start = datetime(2026, 6, 1, 0, 0, 0)

    auth_events: list[AuthEvent] = []
    txn_events: list[TransactionEvent] = []

    # Generate baseline normal activity
    for day in range(NUM_NORMAL_DAYS):
        day_start = start + timedelta(days=day)
        for user in users:
            for kind, event in gen_normal_events(user, day_start):
                if kind == "auth":
                    auth_events.append(event)
                else:
                    txn_events.append(event)

    print(f"Generated {len(auth_events)} normal auth events")
    print(f"Generated {len(txn_events)} normal transactions")

    # Inject attack chains
    attack_labels = []
    target_users = random.sample(users, ATTACK_CHAINS_TO_INJECT)
    for user in target_users:
        attack_day = random.randint(NUM_NORMAL_DAYS - 3, NUM_NORMAL_DAYS - 1) # Attack is injected in the last 3 days of the normal activity period to simulate a recent compromise, which is more realistic and challenging for detection.
        attack_time = start + timedelta(
            days=attack_day,
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
        chain = gen_ato_attack_chain(user, attack_time)
        for kind, event in chain:
            if kind == "auth":
                auth_events.append(event)
            else:
                txn_events.append(event)
                # The transaction in an attack chain is our ground-truth fraud label
                attack_labels.append({
                    "transaction_id": event.event_id,
                    "user_id": user.user_id,
                    "attack_type": "ATO_impossible_travel",
                    "expected_mitre": ["T1110", "T1078"],
                })

    print(f"Injected {ATTACK_CHAINS_TO_INJECT} ATO attack chains")
    print(f"Total auth events: {len(auth_events)}")
    print(f"Total txn events:  {len(txn_events)}")

    # Sort chronologically
    auth_events.sort(key=lambda e: e.timestamp)
    txn_events.sort(key=lambda e: e.timestamp)

    # Write JSONL
    with (OUTPUT_DIR / "auth_events.jsonl").open("w") as f:
        for e in auth_events:
            f.write(json.dumps(asdict(e)) + "\n")
    with (OUTPUT_DIR / "transaction_events.jsonl").open("w") as f:
        for e in txn_events:
            f.write(json.dumps(asdict(e)) + "\n")
    with (OUTPUT_DIR / "attack_labels.json").open("w") as f:
        json.dump(attack_labels, f, indent=2)

    print(f"\nWrote: {OUTPUT_DIR}/auth_events.jsonl")
    print(f"Wrote: {OUTPUT_DIR}/transaction_events.jsonl")
    print(f"Wrote: {OUTPUT_DIR}/attack_labels.json  (ground truth for evaluation)")


if __name__ == "__main__":
    main()