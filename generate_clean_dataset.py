#!/usr/bin/env python3
"""
Generate a clean 100K smart home dataset with consistent labels.
Commands map deterministically to device states — no contradictions.
"""

import csv
import json
import random
import itertools

random.seed(42)

# ============================================================
# House Schema (2BHK)
# ============================================================
ROOMS = ["bathroom", "bedroom", "balcony", "dining_room", "hall", "kitchen", "living_room", "study_room"]

ROOM_DEVICES = {
    "bathroom":    ["exhaust", "geyser", "lights"],
    "bedroom":     ["ac", "blinds", "fan", "lights"],
    "balcony":     ["blinds", "lights"],
    "dining_room": ["ac", "fan", "lights"],
    "hall":        ["lights", "tv"],
    "kitchen":     ["exhaust", "fan", "lights"],
    "living_room": ["fan", "lights", "music_system", "tv"],
    "study_room":  ["computer", "fan", "lights"],
}

# Device-specific values
DEVICE_ON_VALUES = {
    "fan": ["on"],
    "lights": ["on", "bright", "dim", "warm"],
    "tv": ["on"],
    "ac": ["on", "18°C", "19°C", "20°C", "21°C", "22°C", "23°C", "24°C", "25°C"],
    "blinds": ["open"],
    "exhaust": ["on"],
    "geyser": ["on"],
    "computer": ["on"],
    "music_system": ["soft", "medium", "loud", "party", "relax"],
}

DEVICE_OFF_VALUES = {
    "fan": ["off"],
    "lights": ["off"],
    "tv": ["off"],
    "ac": ["off"],
    "blinds": ["close"],
    "exhaust": ["off"],
    "geyser": ["off"],
    "computer": ["off"],
    "music_system": ["off"],
}

# ============================================================
# Command Templates
# ============================================================

# --- Explicit per-device commands (turn on/off specific device in room) ---
TURN_ON_TEMPLATES = [
    "turn on the {device} in the {room}",
    "in the {room} turn on the {device}",
    "switch on the {device} in the {room}",
    "activate the {device} in the {room}",
    "start the {device} in the {room}",
    "Please turn on the {device} in the {room}.",
    "Could you turn on the {device} in the {room}?",
    "Hey, turn on the {device} in the {room} please.",
    "Kindly turn on the {device} in the {room}.",
    "I want the {device} in the {room} on.",
    "in the {room}, switch on the {device}",
    "fire up the {device} in the {room}",
    "enable the {device} in the {room}",
    "power on the {device} in the {room}",
    "Hey home, turn on the {device} in the {room}.",
    "Can you activate the {device} in the {room}?",
    "Set the {device} to on in the {room}.",
]

TURN_OFF_TEMPLATES = [
    "turn off the {device} in the {room}",
    "in the {room} turn off the {device}",
    "switch off the {device} in the {room}",
    "deactivate the {device} in the {room}",
    "stop the {device} in the {room}",
    "Please turn off the {device} in the {room}.",
    "Could you turn off the {device} in the {room}?",
    "Hey, turn off the {device} in the {room} please.",
    "Kindly turn off the {device} in the {room}.",
    "I want the {device} in the {room} off.",
    "in the {room}, switch off the {device}",
    "shut off the {device} in the {room}",
    "disable the {device} in the {room}",
    "power off the {device} in the {room}",
    "Hey home, turn off the {device} in the {room}.",
    "Can you deactivate the {device} in the {room}?",
    "Set the {device} to off in the {room}.",
]

# --- AC-specific with temperature ---
AC_TEMP_TEMPLATES = [
    "set the ac to {temp} in the {room}",
    "in the {room}, set ac to {temp}",
    "turn on the ac to {temp} in the {room}",
    "Please set the ac to {temp} in the {room}.",
    "Hey, set the {room} ac to {temp}.",
    "adjust the ac to {temp} in the {room}",
    "Could you set the ac in the {room} to {temp}?",
    "I want the ac at {temp} in the {room}.",
]

# --- Lights-specific with mode ---
LIGHTS_MODE_TEMPLATES = [
    "set the lights to {mode} in the {room}",
    "in the {room}, set lights to {mode}",
    "turn the lights to {mode} in the {room}",
    "Please set {room} lights to {mode}.",
    "Hey, make the {room} lights {mode}.",
    "adjust the lights to {mode} in the {room}",
    "Could you set the lights in the {room} to {mode}?",
    "I want {mode} lights in the {room}.",
    "switch the {room} lights to {mode}",
]

# --- Music system with mode ---
MUSIC_MODE_TEMPLATES = [
    "set the music_system to {mode} in the {room}",
    "play {mode} music in the {room}",
    "in the {room}, set music to {mode}",
    "Hey, play some {mode} music in the {room}.",
    "Please set the music_system to {mode} in the {room}.",
    "Could you put {mode} music on in the {room}?",
    "I want {mode} music in the {room}.",
]

# --- Blinds-specific ---
BLINDS_OPEN_TEMPLATES = [
    "open the blinds in the {room}",
    "in the {room}, open the blinds",
    "Please open the {room} blinds.",
    "Could you open the blinds in the {room}?",
    "Hey, open the blinds in the {room} please.",
]

BLINDS_CLOSE_TEMPLATES = [
    "close the blinds in the {room}",
    "in the {room}, close the blinds",
    "Please close the {room} blinds.",
    "Could you close the blinds in the {room}?",
    "Hey, close the blinds in the {room} please.",
]

# --- Scene/mood commands (multi-device) ---
SCENE_COMMANDS = [
    # Sleep mode
    {
        "templates": [
            "Good night, set up sleep mode.",
            "Hey home, sound sleep for tonight, thanks.",
            "Time to sleep, goodnight mode please.",
            "Set up sleep mode please.",
            "Activate night mode.",
            "Going to bed, goodnight.",
            "Hey home, bedtime mode please.",
            "Set everything for sleep.",
            "Night mode on please.",
            "I'm going to sleep, set up the house.",
        ],
        "output": {
            "bedroom": {"ac": "22°C", "blinds": "close", "lights": "off", "fan": "on"},
            "living_room": {"lights": "off", "tv": "off"},
            "hall": {"lights": "off"},
            "kitchen": {"lights": "off"},
        }
    },
    # Morning mode
    {
        "templates": [
            "Good morning, wake up mode.",
            "Hey home, morning routine please.",
            "Set up morning mode.",
            "Activate wake up mode.",
            "Rise and shine mode.",
            "Hey home, good morning!",
            "Start morning mode please.",
            "Wake up mode on.",
            "Morning, set up the house.",
            "Good morning, open everything up.",
        ],
        "output": {
            "bedroom": {"blinds": "open", "lights": "bright", "fan": "off"},
            "kitchen": {"lights": "on", "exhaust": "on"},
            "bathroom": {"geyser": "on", "lights": "on"},
        }
    },
    # Movie mode
    {
        "templates": [
            "Set up movie mode in the living room.",
            "Hey home, movie night!",
            "Activate movie mode.",
            "Time for a movie, set up the room.",
            "Movie night mode please.",
            "Set everything for movie watching.",
            "Hey, let's watch a movie.",
            "Cinema mode on.",
            "Movie mode please.",
            "Set up the living room for a movie.",
        ],
        "output": {
            "living_room": {"tv": "on", "lights": "dim", "fan": "on"},
            "hall": {"lights": "off"},
        }
    },
    # Party mode
    {
        "templates": [
            "Party mode on!",
            "Set up for a party.",
            "Hey home, it's party time!",
            "Activate party mode.",
            "Let's party, set up the house.",
            "Party mode please.",
            "Turn on party mode.",
            "Set everything for party.",
            "Hey, party setup!",
            "Time to party, set up the house.",
        ],
        "output": {
            "living_room": {"lights": "bright", "music_system": "party", "tv": "on", "fan": "on"},
            "dining_room": {"lights": "bright"},
            "hall": {"lights": "bright"},
            "kitchen": {"lights": "on"},
        }
    },
    # Study/focus mode
    {
        "templates": [
            "Set up study mode.",
            "Hey home, focus mode please.",
            "Activate study mode.",
            "Time to study, set up the room.",
            "Focus mode on.",
            "Could you please focus mode for tonight now.",
            "Set up for studying.",
            "I need to concentrate, focus mode.",
            "Study mode please.",
            "Set up the study room for work.",
        ],
        "output": {
            "study_room": {"computer": "on", "lights": "bright", "fan": "on"},
            "living_room": {"tv": "off", "music_system": "off"},
            "bedroom": {"lights": "off"},
        }
    },
    # Leaving home
    {
        "templates": [
            "I'm leaving, turn everything off.",
            "Hey home, I'm going out.",
            "Leaving home mode.",
            "Turn off everything, I'm leaving.",
            "Set up away mode.",
            "Going out, shut everything down.",
            "Away mode please.",
            "I'm heading out, turn off the house.",
            "Goodbye mode on.",
            "Leaving the house, secure everything.",
        ],
        "output": {
            "bedroom": {"ac": "off", "lights": "off", "fan": "off"},
            "living_room": {"tv": "off", "lights": "off", "fan": "off", "music_system": "off"},
            "kitchen": {"lights": "off", "exhaust": "off", "fan": "off"},
            "hall": {"lights": "off", "tv": "off"},
            "bathroom": {"lights": "off", "geyser": "off", "exhaust": "off"},
            "study_room": {"computer": "off", "lights": "off", "fan": "off"},
            "dining_room": {"lights": "off", "fan": "off", "ac": "off"},
            "balcony": {"lights": "off"},
        }
    },
    # Cooking mode
    {
        "templates": [
            "Set up the kitchen for cooking.",
            "Hey home, cooking mode.",
            "Activate cooking mode.",
            "Time to cook, set up the kitchen.",
            "Cooking mode on please.",
            "I'm going to cook, prepare the kitchen.",
            "Kitchen setup for cooking.",
            "Start cooking mode.",
            "Hey, I'm cooking.",
            "Set up for cooking please.",
        ],
        "output": {
            "kitchen": {"lights": "bright", "exhaust": "on", "fan": "on"},
        }
    },
    # Relax mode
    {
        "templates": [
            "Set up relax mode.",
            "Hey home, I want to relax.",
            "Activate relaxation mode.",
            "Time to unwind.",
            "Relax mode on.",
            "Set the house for relaxing.",
            "Chill mode please.",
            "I want to relax, set up the house.",
            "Unwind mode on.",
            "Hey, relax mode please.",
        ],
        "output": {
            "living_room": {"lights": "warm", "music_system": "relax", "fan": "on"},
            "bedroom": {"ac": "23°C", "lights": "dim"},
        }
    },
    # Work from home
    {
        "templates": [
            "Set up work from home mode.",
            "Hey home, WFH mode.",
            "Working from home today.",
            "Start work mode.",
            "Activate WFH mode.",
            "Set up start work mode please.",
            "Hey, working from home.",
            "Office mode on.",
            "Home office setup.",
            "Set up for remote work.",
        ],
        "output": {
            "study_room": {"computer": "on", "lights": "bright", "fan": "on"},
            "kitchen": {"lights": "on"},
            "dining_room": {"ac": "23°C"},
        }
    },
    # Dinner mode
    {
        "templates": [
            "Set up for dinner.",
            "Hey home, dinner time!",
            "Activate dinner mode.",
            "Dinner mode please.",
            "Set the dining room for dinner.",
            "Time for dinner, set up.",
            "Dinner setup please.",
            "Hey, we're having dinner.",
            "Set up dinner mode.",
            "Dinner time, prepare the house.",
        ],
        "output": {
            "dining_room": {"lights": "warm", "ac": "23°C", "fan": "on"},
            "kitchen": {"lights": "on", "exhaust": "on"},
            "living_room": {"music_system": "soft"},
        }
    },
]

# --- Multi-device joiners ---
JOINERS = [" and ", " & ", ", then ", "; ", ", also "]
PREFIXES = [
    "", "Please ", "Hey, ", "Could you ", "Kindly ", "Hey home, ",
    "Can you ", "I want to ", "I need to ", "",
]
SUFFIXES = [
    "", " please.", " now.", " thanks.", ".", " please. now.",
    ", thanks.", " please!", "",
]


def generate_single_device_command():
    """Generate a command for a single device in a single room."""
    room = random.choice(ROOMS)
    device = random.choice(ROOM_DEVICES[room])
    
    # Decide on/off/specific value
    action = random.choice(["on", "off", "specific"])
    
    if action == "on" or (action == "specific" and device not in ["ac", "lights", "music_system", "blinds"]):
        value = random.choice(DEVICE_ON_VALUES[device])
        if device == "blinds":
            template = random.choice(BLINDS_OPEN_TEMPLATES)
            cmd = template.format(room=room)
        elif device == "lights" and value in ["bright", "dim", "warm"]:
            template = random.choice(LIGHTS_MODE_TEMPLATES)
            cmd = template.format(room=room, mode=value)
        elif device == "ac" and value != "on" and value.endswith("°C"):
            template = random.choice(AC_TEMP_TEMPLATES)
            cmd = template.format(room=room, temp=value)
        elif device == "music_system" and value != "on":
            template = random.choice(MUSIC_MODE_TEMPLATES)
            cmd = template.format(room=room, mode=value)
        else:
            template = random.choice(TURN_ON_TEMPLATES)
            cmd = template.format(room=room, device=device)
    elif action == "specific":
        # Specific values for special devices
        if device == "ac":
            temp = random.choice(["18°C", "19°C", "20°C", "21°C", "22°C", "23°C", "24°C", "25°C"])
            value = temp
            template = random.choice(AC_TEMP_TEMPLATES)
            cmd = template.format(room=room, temp=temp)
        elif device == "lights":
            mode = random.choice(["bright", "dim", "warm"])
            value = mode
            template = random.choice(LIGHTS_MODE_TEMPLATES)
            cmd = template.format(room=room, mode=mode)
        elif device == "music_system":
            mode = random.choice(["soft", "medium", "loud", "party", "relax"])
            value = mode
            template = random.choice(MUSIC_MODE_TEMPLATES)
            cmd = template.format(room=room, mode=mode)
        elif device == "blinds":
            value = "open"
            template = random.choice(BLINDS_OPEN_TEMPLATES)
            cmd = template.format(room=room)
        else:
            value = "on"
            template = random.choice(TURN_ON_TEMPLATES)
            cmd = template.format(room=room, device=device)
    else:  # off
        value = random.choice(DEVICE_OFF_VALUES[device])
        if device == "blinds":
            template = random.choice(BLINDS_CLOSE_TEMPLATES)
            cmd = template.format(room=room)
        else:
            template = random.choice(TURN_OFF_TEMPLATES)
            cmd = template.format(room=room, device=device)
    
    output = {room: {device: value}}
    return cmd, output


def generate_multi_device_command():
    """Generate a command controlling multiple devices across rooms."""
    num_actions = random.choice([2, 2, 2, 3, 3, 4])
    
    # Pick rooms and devices
    actions = []
    output = {}
    used = set()
    
    for _ in range(num_actions):
        room = random.choice(ROOMS)
        device = random.choice(ROOM_DEVICES[room])
        key = (room, device)
        if key in used:
            continue
        used.add(key)
        
        action = random.choice(["on", "off", "specific"])
        
        if action == "off":
            value = random.choice(DEVICE_OFF_VALUES[device])
            if device == "blinds":
                template = random.choice(BLINDS_CLOSE_TEMPLATES)
                cmd_part = template.format(room=room)
            else:
                template = random.choice(TURN_OFF_TEMPLATES)
                cmd_part = template.format(room=room, device=device)
        elif action == "specific" and device in ["ac", "lights", "music_system"]:
            if device == "ac":
                value = random.choice(["18°C", "19°C", "20°C", "21°C", "22°C", "23°C", "24°C", "25°C"])
                template = random.choice(AC_TEMP_TEMPLATES)
                cmd_part = template.format(room=room, temp=value)
            elif device == "lights":
                value = random.choice(["bright", "dim", "warm"])
                template = random.choice(LIGHTS_MODE_TEMPLATES)
                cmd_part = template.format(room=room, mode=value)
            else:
                value = random.choice(["soft", "medium", "loud", "party", "relax"])
                template = random.choice(MUSIC_MODE_TEMPLATES)
                cmd_part = template.format(room=room, mode=value)
        else:
            value = random.choice(DEVICE_ON_VALUES[device])
            if device == "blinds":
                template = random.choice(BLINDS_OPEN_TEMPLATES)
                cmd_part = template.format(room=room)
            elif device == "lights" and value in ["bright", "dim", "warm"]:
                template = random.choice(LIGHTS_MODE_TEMPLATES)
                cmd_part = template.format(room=room, mode=value)
            elif device == "ac" and value.endswith("°C"):
                template = random.choice(AC_TEMP_TEMPLATES)
                cmd_part = template.format(room=room, temp=value)
            elif device == "music_system" and value not in ["on"]:
                template = random.choice(MUSIC_MODE_TEMPLATES)
                cmd_part = template.format(room=room, mode=value)
            else:
                template = random.choice(TURN_ON_TEMPLATES)
                cmd_part = template.format(room=room, device=device)
        
        actions.append(cmd_part)
        if room not in output:
            output[room] = {}
        output[room][device] = value
    
    if len(actions) < 2:
        return generate_single_device_command()
    
    # Join actions
    joiner = random.choice(JOINERS)
    cmd = joiner.join(actions)
    
    prefix = random.choice(PREFIXES)
    suffix = random.choice(SUFFIXES)
    cmd = prefix + cmd + suffix
    
    return cmd, output


def generate_scene_command():
    """Generate a scene/mood command."""
    scene = random.choice(SCENE_COMMANDS)
    template = random.choice(scene["templates"])
    output = scene["output"]
    return template, output


def main():
    output_path = "smart_home_100k_clean.csv"
    
    # Distribution: 40% single device, 45% multi device, 15% scene
    num_single = 40000
    num_multi = 45000
    num_scene = 15000
    total = num_single + num_multi + num_scene
    
    print(f"Generating {total} examples...")
    
    rows = []
    
    print(f"  Generating {num_single} single-device commands...")
    for _ in range(num_single):
        cmd, output = generate_single_device_command()
        # Sort output keys for consistency
        sorted_output = {k: (dict(sorted(v.items())) if isinstance(v, dict) else v) 
                        for k, v in sorted(output.items())}
        rows.append({"input": cmd, "output": json.dumps(sorted_output)})
    
    print(f"  Generating {num_multi} multi-device commands...")
    for _ in range(num_multi):
        cmd, output = generate_multi_device_command()
        sorted_output = {k: (dict(sorted(v.items())) if isinstance(v, dict) else v) 
                        for k, v in sorted(output.items())}
        rows.append({"input": cmd, "output": json.dumps(sorted_output)})
    
    print(f"  Generating {num_scene} scene commands...")
    for _ in range(num_scene):
        cmd, output = generate_scene_command()
        sorted_output = {k: (dict(sorted(v.items())) if isinstance(v, dict) else v) 
                        for k, v in sorted(output.items())}
        rows.append({"input": cmd, "output": json.dumps(sorted_output)})
    
    # Shuffle
    random.shuffle(rows)
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["input", "output"])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n✓ Generated {len(rows)} examples → {output_path}")
    
    # Verify consistency
    print("\nVerification:")
    value_map = {}
    for row in rows:
        inp = row["input"].lower()
        out = json.loads(row["output"])
        for room, devs in out.items():
            if not isinstance(devs, dict):
                continue
            for dev, val in devs.items():
                if f"turn on the {dev}" in inp:
                    key = f"turn on {dev}"
                    value_map.setdefault(key, {})
                    value_map[key][val] = value_map[key].get(val, 0) + 1
                elif f"turn off the {dev}" in inp:
                    key = f"turn off {dev}"
                    value_map.setdefault(key, {})
                    value_map[key][val] = value_map[key].get(val, 0) + 1
    
    print("\n  Command → value distribution (should be consistent):")
    for phrase in sorted(value_map.keys())[:15]:
        vals = value_map[phrase]
        total = sum(vals.values())
        print(f"    \"{phrase}\" ({total}x): ", end="")
        for v, c in sorted(vals.items(), key=lambda x: -x[1]):
            print(f"{v}={100*c/total:.0f}% ", end="")
        print()
    
    # Show samples
    print("\n  Sample rows:")
    for row in rows[:5]:
        print(f"    IN:  {row['input']}")
        print(f"    OUT: {row['output']}")
        print()


if __name__ == "__main__":
    main()
