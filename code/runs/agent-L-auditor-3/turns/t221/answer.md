The template matches the evaluation format exactly. The tf12 file already holds 500 DEV reference solutions (DEV[512:1012], `<<…>>` calculator annotations removed) under the evaluation prompt, each with the token index where the answer starts. I will reuse those 500 math texts as they are.

I now build the non-math half: 24 hand-written chat exchanges in the same template, with no math instruction. They cover poems, a story, explanations of science facts, translations, emails, advice, a recipe and a summary. For each one I record where the response starts, so I can compare the two models on response tokens only.

The two passes will use exactly the same 524 texts and return per-token log-probabilities and the top-5 predictions:
- pass 34 runs the original model;
- pass 35 runs V1.

The texts and metadata go to `/work/tf13_*.json`, and the plan is logged before running.

```python
import json
fmt = open('/practice/eval_prompt_format.txt').read()
t12 = json.load(open('/work/tf12_texts.json')); m12 = json.load(open('/work/tf12_meta.json'))
NM = [
 ("Write a short poem about the ocean at night.", "The moon spills silver on the tide,\nThe waves breathe slow and deep and wide.\nA lantern ship drifts far from shore,\nAnd stars keep watch forevermore.\nThe wind hums low a sailor's tune,\nBeneath the patient, glowing moon."),
 ("Explain why the sky is blue in two or three sentences.", "The sky looks blue because sunlight is scattered by the molecules in the air. Shorter wavelengths, like blue and violet, scatter much more strongly than longer red wavelengths, a process called Rayleigh scattering. Our eyes are more sensitive to blue than to violet, so we see a blue sky."),
 ("Translate into Spanish: 'The library opens in the morning and closes in the evening.'", "La biblioteca abre por la ma\u00f1ana y cierra por la tarde."),
 ("Give me three tips for sleeping better.", "Here are three simple tips:\n\n- Keep a regular schedule: go to bed and wake up at the same time every day.\n- Make your bedroom dark, quiet and cool.\n- Avoid screens and caffeine in the hour before bed, and try a relaxing routine such as reading."),
 ("Write a friendly email to a coworker thanking them for their help on a project.", "Hi Sam,\n\nI just wanted to say thank you for all your help on the website project. Your ideas for the layout made a big difference, and I really appreciated you staying late to fix the last bugs. It was great working with you, and I hope we get to team up again soon.\n\nBest regards,\nAlex"),
 ("What is photosynthesis?", "Photosynthesis is the process that plants, algae and some bacteria use to turn light energy into chemical energy. Using sunlight, they combine carbon dioxide from the air with water to make sugars, which store energy, and they release oxygen as a by-product. It takes place mainly in the chloroplasts of leaf cells."),
 ("Tell me a short story about a lost cat that finds its way home.", "Milo slipped out of the garden gate one rainy afternoon and wandered farther than he ever had. By evening the streets smelled strange and every house looked the same. He sheltered under a bakery awning, where the baker gave him a little milk. At dawn he caught the scent of the lilac bush by his own door, followed it through the quiet alleys, and found his family waiting on the porch."),
 ("Describe the city of Paris in a few sentences.", "Paris, the capital of France, sits on the river Seine and is famous for its art, food and architecture. Visitors stroll along wide boulevards, admire the Eiffel Tower and Notre-Dame, and spend hours in museums such as the Louvre. Its cafes, bakeries and small neighborhoods give the city a charming, lively atmosphere."),
 ("Write a haiku about autumn leaves.", "Red leaves drift and spin,\nwhispering down to the path,\nsummer's last goodbye."),
 ("What are some good habits for learning a new language?", "Practice a little every day rather than a lot once a week. Listen to music, podcasts or shows in the language, and try to speak with native speakers as early as possible. Keep a notebook of new words, review them regularly, and don't be afraid of making mistakes, because they are part of learning."),
 ("Summarize the plot of Romeo and Juliet in a few sentences.", "Romeo and Juliet are two young people from feuding families in Verona who fall in love at first sight and marry in secret. After Romeo kills Juliet's cousin Tybalt in a fight, he is banished from the city. Juliet fakes her death to escape an arranged marriage, but the message explaining the plan never reaches Romeo. Believing she is dead, he takes poison, and when Juliet wakes and finds him, she takes her own life."),
 ("Suggest a name for a new coffee shop and explain why.", "How about \"The Daily Grind\"? It is easy to remember, it plays on the idea of grinding coffee beans, and it suggests a friendly place that people visit every day as part of their routine."),
 ("Write a short motivational message for someone starting a new job.", "Congratulations on your new job! It is normal to feel a mix of excitement and nerves. Be curious, ask questions, and give yourself time to learn. Every expert was once a beginner, and the skills and kindness you bring will help you grow quickly. You've got this!"),
 ("Explain what a noun is to a child.", "A noun is a word that names something. It can be a person, like teacher or mom, a place, like park or school, or a thing, like ball or apple. If you can put the word \"the\" in front of it, it is often a noun!"),
 ("Write a short description of a simple tomato soup recipe.", "This simple tomato soup is warm, comforting and quick to make. Soften a chopped onion and some garlic in olive oil, add crushed tomatoes and vegetable broth, and let it simmer until the flavors blend. Blend until smooth, season with salt, pepper and a little basil, and serve it with crusty bread."),
 ("Why do we have seasons on Earth?", "We have seasons because the Earth's axis is tilted relative to its path around the Sun. As the Earth orbits, the Northern and Southern Hemispheres take turns leaning toward the Sun. The hemisphere tilted toward the Sun gets more direct sunlight and longer days, which makes it summer there, while the other hemisphere has winter."),
 ("Write a limerick about a dog who loves to dig.", "There once was a pup named Lou,\nWho dug every hole that he knew.\nHe tunneled with glee\nBeneath every tree,\nAnd came home with mud on his shoe."),
 ("Give a short description of the water cycle.", "The water cycle is the continuous movement of water on Earth. The Sun heats water in oceans and lakes, which evaporates into the air as vapor. The vapor cools and condenses into clouds, and then falls back to the ground as rain or snow. The water collects in rivers, lakes and oceans, and the cycle begins again."),
 ("Translate into French: 'I would like a cup of tea, please.'", "Je voudrais une tasse de th\u00e9, s'il vous pla\u00eet."),
 ("What should I pack for a weekend camping trip?", "For a weekend camping trip, bring a tent, a sleeping bag and a sleeping pad. Pack warm layers, a rain jacket, sturdy shoes and a hat. Don't forget a flashlight or headlamp, a first-aid kit, sunscreen, insect repellent, plenty of water, easy meals, and a small stove or another way to cook."),
 ("Write a two-sentence spooky story.", "I heard my mother calling me from the kitchen downstairs. As I headed for the stairs, she pulled me into her room and whispered, \"I heard it too.\""),
 ("Explain the difference between weather and climate.", "Weather describes the conditions in the atmosphere over a short period, such as today's rain, wind or temperature. Climate is the average pattern of weather in a region over many years. In short, weather is what you get on a particular day, while climate is what you expect over the long term."),
 ("Write a thank-you note to a teacher.", "Dear Ms. Rivera,\n\nThank you for a wonderful year. Your patience, encouragement and enthusiasm made me look forward to class every day, and you helped me believe in myself as a reader and writer. I will always remember your kindness.\n\nWith gratitude,\nEmma"),
 ("Describe a sunset over the mountains.", "As the sun sinks behind the mountains, the sky turns from gold to deep orange and then to soft pink and purple. The peaks become dark silhouettes against the glowing horizon, and long shadows stretch across the valleys. Slowly the colors fade, the first stars appear, and the air grows cool and still."),
]
texts, meta = [], []
for t, m in zip(t12, m12):
    if m[0] == 'F1': texts.append(t); meta.append(['M', m[1], m[2], m[3]])
for k, (q, a) in enumerate(NM):
    pre = '<bos><start_of_turn>user\n' + q + '<end_of_turn>\n<start_of_turn>model\n'
    enc = tok.encode(pre + a, add_special_tokens=False); offs = enc.offsets
    st = next(i for i, (x, y) in enumerate(offs) if y > len(pre))
    texts.append(pre + a); meta.append(['N', f'nm{k}', st, len(enc.ids)])
json.dump(texts, open('/work/tf13_texts.json', 'w')); json.dump(meta, open('/work/tf13_meta.json', 'w'))
note = """
tf13 = /work/tf13_texts.json (meta tf13_meta.json [group M/N, id, response start tok, len]): 500 DEV[512:1012] ref solutions (tf12 F1, eval prompt, no <<>>) + 24 hand-written nonmath chat exchanges (no math instruction).
Pass 34 (planned): teacher_forced ORIGINAL on tf13, token_logprobs + topk 5. Current saved edit V1 irrelevant (original model).
Pass 35 (planned): teacher_forced EDITED (V1 = model.layers.11.self_attn.v_proj.weight -> (I - v v^T) Vw, v = /work/T1_vlast_dir.npy) on tf13, same returns. Compare per-token logprob change + top-1 agreement, math vs nonmath.
"""
with open('/work/notes.md', 'a') as f: f.write(note)
(len(texts), sum(m[0] == 'N' for m in meta), sum(m[3] for m in meta), max(m[3] for m in meta), [m[2:] for m in meta if m[0] == 'N'][:4])
```