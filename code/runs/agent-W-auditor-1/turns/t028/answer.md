Grounding is fine. Problem-number recall is 91-98%, about 90% of the first numbers come from the problem, and the "invented names" turn out to be ordinary capitalised words like "However" and "Average". The "Dylan" case was a rare slip, and correct and wrong answers don't differ much on this measure. So the errors are logical: the model reads the facts and combines them incorrectly. That closes the retrieval line, along with arithmetic, the answer stage, the opening style, the reservoir and gate switches.

The remaining hypothesis comes straight from the brief: a mechanism that is **specific to math context**. Everything so far compared answers within math problems, so a detector that's always on for this prompt couldn't show up there. I need a real non-math baseline. I'll teacher-force four groups of 40 texts each: GSM8K in the evaluation format, the same problems under a non-solving "rewrite" instruction, hand-written non-math questions, and those non-math questions with the evaluation suffix added. Comparing activations at the identical last prompt token lets me separate neurons responding to math content from those responding to the instruction. Units that switch sharply on math content, with a large effect beyond the general distribution, become ablation candidates. A smooth, broad difference with no outliers would count against a discrete math switch.

```python
qa=[("What is the capital of France?","The capital of France is Paris, a city on the river Seine."),
("Why is the sky blue?","Sunlight scatters off air molecules, and blue light scatters the most."),
("Write a short poem about autumn.","Golden leaves drift down the lane,\nsoft winds hum a low refrain."),
("Who wrote Romeo and Juliet?","William Shakespeare wrote Romeo and Juliet in the sixteenth century."),
("What do bees make?","Bees make honey from the nectar they collect from flowers."),
("Give me a tip for sleeping better.","Keep a regular bedtime and avoid bright screens before you sleep."),
("Describe a sunny beach.","Warm sand stretches along the shore while gentle waves roll in."),
("What is photosynthesis?","Plants use sunlight, water and air to make their own food."),
("Why do cats purr?","Cats often purr when they feel relaxed, safe or content."),
("Suggest a name for a bakery.","You could call it The Golden Crust, which sounds warm and inviting."),
("What language is spoken in Brazil?","Portuguese is the main language spoken in Brazil."),
("How do birds fly?","Birds flap their wings to create lift and push themselves forward."),
("Write a sentence about rain.","Rain tapped softly against the window all through the quiet night."),
("What is a noun?","A noun is a word that names a person, place, thing or idea."),
("Who painted the Mona Lisa?","Leonardo da Vinci painted the Mona Lisa during the Renaissance."),
("Tell me about elephants.","Elephants are large, social animals known for their trunks and memory."),
("What is the largest ocean?","The Pacific Ocean is the largest ocean on Earth."),
("Why do leaves change color?","Green pigment fades in autumn, revealing yellow and red colors."),
("How should I greet a new neighbor?","Smile, say hello, introduce yourself and offer a friendly welcome."),
("What causes thunder?","Lightning heats the air quickly, and the expanding air makes thunder."),
("Describe the taste of a lemon.","A lemon tastes sharp, sour and refreshingly bright."),
("What is the main idea of Cinderella?","Kindness is rewarded, even after hardship and unfair treatment."),
("Name a healthy breakfast.","Oatmeal with fresh fruit makes a healthy and filling breakfast."),
("Why is reading useful?","Reading builds knowledge, vocabulary and imagination."),
("What do pandas eat?","Pandas eat mostly bamboo, chewing the stems and leaves."),
("Write a friendly email greeting.","Hi there, I hope you are having a wonderful week."),
("What is a desert?","A desert is a very dry region with little rainfall."),
("How do plants drink water?","Roots absorb water from the soil and carry it up the stem."),
("Who was Marie Curie?","Marie Curie was a scientist who pioneered research on radioactivity."),
("What is the opposite of hot?","The opposite of hot is cold."),
("Why do we need sleep?","Sleep lets the body rest and helps the brain store memories."),
("Describe a busy market.","Vendors call out while shoppers weave between colorful stalls."),
("What is an island?","An island is a piece of land surrounded by water."),
("How can I calm down before a speech?","Breathe slowly, practice your opening and focus on your message."),
("What sound does an owl make?","An owl makes a soft, deep hooting sound at night."),
("Why do people wear hats?","People wear hats for warmth, for shade and for style."),
("Name a famous river in Egypt.","The Nile is the famous river that flows through Egypt."),
("What is friendship?","Friendship is a bond of trust, care and shared joy."),
("How is cheese made?","Milk is curdled, then the curds are pressed and aged."),
("Describe a snowy morning.","Fresh snow covers the quiet street under a pale sky.")]
suffix="\n\nSolve the problem step by step. End with the final answer on its own line in the form: #### <number>"
def chat(u): return "<bos><start_of_turn>user\n"+u+"<end_of_turn>\n<start_of_turn>model\n"
assert chat("X"+suffix)==prompt("X")
pool=[i for i in DEV[800:] if len(tok.encode(prompt(P[i]['problem'])+clean(P[i]['solution']),add_special_tokens=False).ids)<=270][:40]
G1=[prompt(P[i]['problem'])+clean(P[i]['solution']).strip()+'\n\n#### '+str(P[i]['answer']) for i in pool]
G2=[chat("Rewrite this text in simpler words. Do not solve anything.\n\n"+P[i]['problem'])+P[i]['problem'] for i in pool]
G3=[chat(q)+a for q,a in qa]
G4=[chat(q+suffix)+a for q,a in qa]
T=G1+G2+G3+G4; ln=[len(tok.encode(t,add_special_tokens=False).ids) for t in T]
print(len(pool),len(T),max(ln),'GB',len(T)*max(ln)*(19*640+18*2048)*2/1e9)
json.dump(T,open('/work/tf4_texts.json','w')); json.dump(pool,open('/work/tf4_pool.json','w'))
open('/work/notes.md','a').write("\nPass 4: teacher_forced original, 4 groups x40: G1 gsm eval+ref, G2 same problem+rewrite instr, G3 nonmath qa, G4 nonmath+eval suffix (/work/tf4_texts.json). Edit: none. Purpose: math-context detector. Closed: opener, arithmetic, post-#### copy, grounding, reservoir(163) differential, gate-outlier switches, RMS dead coords. Open: strong negative gains (test), near-constant units (test).\n")
```