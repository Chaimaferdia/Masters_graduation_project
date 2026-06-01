from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Teacher:
    id: int
    name: str
    subject: str
    group_hours: Dict
    available_slots: set


@dataclass
class Room:
    id: int
    name: str
    type: str


@dataclass
class StudentGroup:
    id: int
    name: str
    level: int
    stream: str
    room_id: int


@dataclass
class Course:
    id: int
    subject: str
    teacher_id: int
    group_id: int
    duration: int
    stream: str
    room_type_required: str
    merged_group_ids: List[int]