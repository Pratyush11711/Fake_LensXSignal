import urllib.request, json

samples = [
    ("User article", "Well conceptualised driving questions help to improve the effectiveness of project-based learning. Here, ACER India researchers Dimple Sood and Dr Ashu Kapoor share the guiding principles for designing driving questions, based on their recent research."),
    ("Fake sample", "BREAKING: Secret video reveals corrupt officials caught admitting to hidden algorithms altering ballots in key counties! Watch the featured image evidence mainstream media is blacking out — share before it gets deleted!"),
    ("Reuters real", "WASHINGTON (Reuters) - Senate Republicans and Democrats reached a bipartisan compromise on federal budget appropriations late Wednesday, averting a government shutdown before the midnight deadline.")
]

for name, text in samples:
    req = urllib.request.Request('http://127.0.0.1:5000/predict', 
        data=json.dumps({'text': text}).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    res = urllib.request.urlopen(req)
    data = json.loads(res.read().decode('utf-8'))
    print(f"=== {name} ===")
    print(f"  Prediction: {data['prediction']} ({data['confidence']}%)")
    print(f"  Model: {data.get('model')}")
