"""
Generate raw data for the memory scoring pipeline.

Output format:
    memory_db.parquet — flat list of all memory segments
        memory_id : int
        video_id  : str
        text      : str

    qa_db.parquet — question-answer pairs per video
        qa_id    : int
        video_id : str
        question : str
        answer   : str

Feed both files into memory_scorer.py:
  scorer scores ALL memories per video against each question,
  then builds training samples from the scored pool.
"""

import os
import pandas as pd

SCENARIOS = [
    {
        "video_id": "v001",
        "question": "What was the first thing the chef did after entering the kitchen?",
        "answer": "He washed his hands and put on an apron.",
        "memories": {
            "high": [
                "The chef walks into the kitchen, immediately heads to the sink, washes his hands thoroughly, then ties an apron around his waist before approaching the counter.",
                "Camera follows the chef through the kitchen entrance; he pauses at the handwashing station, scrubs for about 20 seconds, dries his hands, and pulls an apron from the hook.",
            ],
            "medium": [
                "The chef is shown preparing ingredients on the counter, chopping vegetables with a large knife.",
                "An overhead shot shows the kitchen layout — sink on the left, stove in the center, refrigerator at the back.",
            ],
            "low": [
                "The host introduces today's episode, explaining that the show focuses on professional kitchen techniques.",
                "A brief montage of the restaurant's exterior and dining area plays before the kitchen segment begins.",
                "The sous-chef is seen seasoning a sauce on the far side of the kitchen.",
                "Background music transitions as the segment moves to a dessert preparation scene.",
                "A sponsor message plays: 'Today's episode is supported by...'",
            ],
        },
    },
    {
        "video_id": "v002",
        "question": "Why did the engineer stop the assembly line during the inspection?",
        "answer": "A misaligned component was detected by the sensor.",
        "memories": {
            "high": [
                "The quality sensor on line 3 flashes red. The engineer immediately presses the emergency stop button and walks over to inspect the conveyor, pointing out a visibly misaligned bracket.",
                "On-screen text reads: 'Line halt triggered — component misalignment detected at station 7.' The engineer confirms the fault and orders the line stopped.",
            ],
            "medium": [
                "The engineer walks along the assembly line, pausing at each station to check the readout screens.",
                "A close-up of the sensor panel shows several green lights and one amber warning indicator.",
            ],
            "low": [
                "Workers on the morning shift are shown clocking in and putting on safety gear.",
                "A time-lapse of the factory floor shows dozens of units rolling off the line over several hours.",
                "The narrator explains the general purpose of the quality inspection system.",
                "A different engineer on line 1 signs off a completed batch without issues.",
                "The factory manager is interviewed about production targets for the quarter.",
            ],
        },
    },
    {
        "video_id": "v003",
        "question": "What did the athlete say immediately after crossing the finish line?",
        "answer": "She said 'I can't believe it' and started crying.",
        "memories": {
            "high": [
                "Slow-motion replay: the athlete breaks the tape, staggers forward, covers her mouth, and says clearly into the trackside microphone, 'I can't believe it,' before tears start streaming down her face.",
                "The post-race interview clip shows the athlete saying, 'When I crossed the line, the first thing out of my mouth was just — I can't believe it. I was already crying before I even stopped moving.'",
            ],
            "medium": [
                "The athlete's coach rushes onto the track and embraces her seconds after she finishes.",
                "The scoreboard updates with the athlete's official time, showing a new personal best.",
            ],
            "low": [
                "Pre-race footage shows the athlete warming up in the tunnel before the start.",
                "The crowd in the stadium is shown cheering as the race begins.",
                "A rival athlete crosses the finish line several seconds later, looking disappointed.",
                "The medal ceremony footage shows all three podium athletes receiving their awards.",
                "Commentary discusses the weather conditions and track surface quality.",
            ],
        },
    },
    {
        "video_id": "v004",
        "question": "At what point in the lecture did the professor introduce the second hypothesis?",
        "answer": "After finishing the experiment results discussion, around the 35-minute mark.",
        "memories": {
            "high": [
                "The lecture timestamp reads 35:12. The professor says, 'Now that we've covered the experimental results, let me move on to the second hypothesis,' and writes 'H2' on the whiteboard.",
                "Slide 18 appears on screen titled 'Hypothesis 2', immediately following a slide showing graphs of experimental data. The professor checks his watch and notes they are about halfway through.",
            ],
            "medium": [
                "The professor is mid-explanation of the experimental setup, pointing to equipment diagrams on the projected slide.",
                "Students are seen taking notes as the professor summarizes the first set of results.",
            ],
            "low": [
                "The lecture begins with the professor greeting students and outlining the day's agenda.",
                "A student in the front row raises a hand and asks a question about the first hypothesis.",
                "The camera briefly shows the back of the auditorium, which is about two-thirds full.",
                "The professor pauses to take a sip of water between sections.",
                "End-of-lecture footage shows students packing up their bags.",
            ],
        },
    },
    {
        "video_id": "v005",
        "question": "What happened to the suspect just before the police arrived?",
        "answer": "He attempted to hide the bag under the bench.",
        "memories": {
            "high": [
                "CCTV footage timestamp 14:23:41 — the suspect glances around nervously, crouches down, and shoves a dark duffel bag beneath the wooden bench. He stands up and walks away just as a police siren is audible in the distance.",
                "The detective's narration: 'The footage clearly shows the subject concealing the bag under the bench at 14:23, approximately 90 seconds before officers arrived on scene.'",
            ],
            "medium": [
                "Earlier footage at 14:15 shows the suspect entering the park carrying the bag visibly.",
                "A witness statement plays: 'I saw a man acting nervously near the benches, kept looking over his shoulder.'",
            ],
            "low": [
                "Officers are shown arriving at the park entrance at 14:25 and fanning out across the area.",
                "A park maintenance worker is interviewed; he says he didn't notice anything unusual.",
                "Wide shot of the park showing families and joggers going about their day.",
                "A different person sits on the same bench earlier in the afternoon, unrelated to the incident.",
                "The evidence bag containing the recovered duffel is shown in the police inventory log.",
            ],
        },
    },
    {
        "video_id": "v006",
        "question": "Which step did the surgeon skip during the procedure, according to the review board?",
        "answer": "The pre-incision timeout confirmation was not performed.",
        "memories": {
            "high": [
                "The review board chairman states: 'Upon reviewing the OR recording, it is evident that the pre-incision timeout — mandatory verification of patient identity, procedure, and site — was not conducted before the first incision was made.'",
                "OR footage: the surgical team proceeds directly from draping to incision without the standard pause for the timeout checklist. No verbal confirmation is heard on the audio track.",
            ],
            "medium": [
                "An anesthesiologist is shown completing her pre-op checklist, but the camera cuts away before the surgical team's joint verification step.",
                "A hospital administrator explains the general purpose of surgical timeouts in patient safety protocols.",
            ],
            "low": [
                "The patient is shown being wheeled into the operating room before the procedure.",
                "Post-operative care footage shows the patient in recovery, stable condition noted.",
                "The hospital's legal team is interviewed about the review process.",
                "A different surgical team in another OR is shown correctly completing their pre-incision timeout.",
                "General footage of the hospital hallways and nursing stations.",
            ],
        },
    },
    {
        "video_id": "v007",
        "question": "What was shown on the dashboard screen when the driver lost control?",
        "answer": "A traction control warning and a speed reading of 142 km/h.",
        "memories": {
            "high": [
                "Dashcam footage freeze-framed at the moment of skid: the instrument cluster clearly shows 142 km/h on the speedometer and an illuminated traction control warning icon — an orange car with wavy lines beneath it.",
                "The accident reconstructionist points to the dashboard screenshot: 'At the instant of loss of control, we have 142 kilometers per hour and a traction control fault active — both visible here.'",
            ],
            "medium": [
                "Earlier dashcam footage shows normal driving at highway speeds, dashboard readings in the 110–120 km/h range.",
                "A mechanic explains that the traction control system on this model activates a warning before fully cutting in.",
            ],
            "low": [
                "Footage of the road conditions that day shows wet asphalt and reduced visibility.",
                "Witnesses at the scene describe hearing a loud screeching sound before the crash.",
                "The driver's phone records are shown; no calls or messages were active at the time.",
                "A police officer measures skid marks on the road.",
                "The vehicle is shown being loaded onto a flatbed truck for examination.",
            ],
        },
    },
    {
        "video_id": "v008",
        "question": "What did the project manager announce at the start of the Monday meeting?",
        "answer": "The client deadline had been moved up by two weeks.",
        "memories": {
            "high": [
                "Meeting recording, 09:02 AM Monday: the project manager opens by saying, 'I have an important update — the client called Friday and they need delivery two weeks earlier than planned. That changes everything on our timeline.'",
                "A team member's notes, shown on screen, read: 'PM announced at start of Monday standup: client deadline moved forward by 2 weeks. Need to reprioritize sprint.'",
            ],
            "medium": [
                "The project manager is seen pulling up a revised Gantt chart on the shared screen shortly after the meeting begins.",
                "Team members exchange concerned looks as the new timeline appears on the projected slides.",
            ],
            "low": [
                "Footage of team members arriving at the office and making coffee before the meeting.",
                "The project manager's end-of-meeting summary covers action items and owners.",
                "A separate Thursday meeting shows a different topic — code review feedback — being discussed.",
                "The office environment is shown: open floor plan, whiteboards covered in sticky notes.",
                "A team member is interviewed later about how they felt about the schedule change.",
            ],
        },
    },
]


def generate_memory_db() -> pd.DataFrame:
    """Flat table of all memory segments across all videos."""
    rows = []
    mid = 0
    for scenario in SCENARIOS:
        for level, segments in scenario["memories"].items():
            for text in segments:
                rows.append({
                    "memory_id": mid,
                    "video_id": scenario["video_id"],
                    "text": text,
                })
                mid += 1
    return pd.DataFrame(rows)


def generate_qa_db() -> pd.DataFrame:
    """One QA pair per video."""
    rows = []
    for i, scenario in enumerate(SCENARIOS):
        rows.append({
            "qa_id": i,
            "video_id": scenario["video_id"],
            "question": scenario["question"],
            "answer": scenario["answer"],
        })
    return pd.DataFrame(rows)


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "raw")
    os.makedirs(output_dir, exist_ok=True)

    memory_db = generate_memory_db()
    qa_db = generate_qa_db()

    memory_db.to_parquet(os.path.join(output_dir, "memory_db.parquet"), index=False)
    qa_db.to_parquet(os.path.join(output_dir, "qa_db.parquet"), index=False)

    print(f"Generated memory_db: {len(memory_db)} segments across {memory_db['video_id'].nunique()} videos")
    print(f"Generated qa_db   : {len(qa_db)} QA pairs")

    print(f"\n[memory_db sample — v001]")
    for _, row in memory_db[memory_db["video_id"] == "v001"].iterrows():
        print(f"  [{row['memory_id']}] {row['text'][:80]}...")

    print(f"\n[qa_db]")
    for _, row in qa_db.iterrows():
        print(f"  {row['video_id']}  Q: {row['question'][:60]}...")


if __name__ == "__main__":
    main()
