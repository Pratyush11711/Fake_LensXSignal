import sys
sys.path.insert(0, '.')
from src.roberta_detector import RobertaFakeNewsDetector

det = RobertaFakeNewsDetector()
print("Detector ready:", det.is_ready)

samples = [
    ("User article (education)", "Well conceptualised driving questions help to improve the effectiveness of project-based learning. Here, ACER India researchers Dimple Sood and Dr Ashu Kapoor share the guiding principles for designing driving questions, based on their recent research."),
    ("Reuters real news", "WASHINGTON (Reuters) - Senate Republicans and Democrats reached a bipartisan compromise on federal budget appropriations late Wednesday, averting a government shutdown before the midnight deadline. The bill passed 74 to 24 and now heads to the White House for presidential signature."),
    ("Fake clickbait bombshell", "BREAKING BOMBSHELL: Secret video reveals corrupt officials caught admitting to hidden algorithms altering ballots in key counties! Watch the featured image evidence mainstream media is blacking out — share before it gets deleted by the deep state cabal!"),
    ("BBC Space real news", "NASA has successfully launched its new Europa Clipper spacecraft on a voyage to explore Jupiter's moon Europa and search for signs of life. The mission will take six years to reach Jupiter.")
]

for name, text in samples:
    res = det.predict(text)
    print(f"=== {name} ===")
    print(f"  Prediction: {res['prediction']} ({res['confidence']}%)")
    print(f"  Probabilities: Real={res['real_probability']*100:.2f}%, Fake={res['fake_probability']*100:.2f}%")
