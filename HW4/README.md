# NLP HW4

Please find the homework assignment instructions [here](https://docs.google.com/document/d/1K8s_Ecms0cIqRO1PKPFs2bfFVFfZpc1nFoEhtxRlCaM/edit?tab=t.vzzfyyxeyrok).

## Part 1

1. Greedy search outputs:

```
Context 1:
Rick grew up in a troubled household. He was a high school dropout, and he was a high school dropout. He was a high school dropout. He was a high school dropout. He was a high school dropout

Context 2:
Laverne needs to prepare something for her friend's party.

"I'm not going to be able to do that," she says. "I'm not going to be able to do that. I'm not going to be able to do that.

Context 3:
Sarah had been dreaming of visiting Europe for years.

"I was so excited to see the world, but I was also so scared of what I was going to see," she said.

"I was so scared of what I was

Context 4:
Gina was worried the cookie dough in the tube would be gross.

"I was like, 'Oh my God, I'm going to have to eat this,'" she said. "I was like, 'I'm going to have to eat this.' I

Context 5:
It was  my final performance in marching band. I was so happy to be back in the band. I was so happy to be back in the band. I was so happy to be back in the band. I was so happy to be back

Context 6:
Jim found an old disposable camera in the bottom of his junk drawer. He took it out and took a picture of it. He took a picture of it and took a picture of it and took a picture of it and took a picture of it and took a picture of

Context 7:
Ron started his new job as a landscaper today.

"I'm a big fan of the outdoors," he said. "I love the outdoors. I love the outdoors. I love the outdoors. I love the outdoors. I love the outdoors

Context 8:
John and Billy became very skilled at beer pong. They were also very good at playing the piano.

"I was a little bit nervous about the music, but I was able to play it. I was able to play it for a while

Context 9:
Caroline was a student in medical school. She was a member of the medical staff at the hospital. She was a member of the medical staff at the hospital. She was a member of the medical staff at the hospital. She was a member

Context 10:
Trish hated the outdoors. She was a little bit of a outdoorsy person, but she was also a little bit of a outdoorsy person. She was a little bit of a outdoorsy person. She was a little bit


```

2. Free response about greedy search:

The greedy search does a really good job of writing plauable, on topic continuations of the prompts for the first sentence. This is because greedy search always picks the highest probably next word making the responses super plausable. The drawback of this method is a lack of creativity as the model is very deterministic. An even larger problem for the greedy search that is not present in even very limited top-k and top-p models, is the output getting stuck in a loop. All of the greedy search options either get stuck in a loop of the same phrase or look like they are beginning to get stuck in a loop. This happens when the most probably words in a given string keep leading to the others to be suggested. Since there is no variability in greedy search where the most probable wrod will not be picked, this will continue forever. Overall, this model is good at finding a very plausable continuation, but needs more variation to not get stuck in a loop.

3. Ancestral sampling outputs:
```
Context 1:
Rick grew up in a troubled household. He seems corrupt, suspicious, and especially reckless. But the factors that account for his unusual behavior are more clear and deeper than his record suggests, suggesting he's one of a single group who may have

Context 2:
Laverne needs to prepare something for her friend's party. Grifled into the truest of short steps, she lets out a tall catch of breath in her mouth that will only cause even the most unaware their familiar passage to pass for one full easily forced are

Context 3:
Sarah had been dreaming of visiting Europe for years. """

Gedaufar Glennist asked, "Well, [she] loved Japan."

"I'm not sure she was buying food at that time," Henry responded.



Context 4:
Gina was worried the cookie dough in the tube would be gross.

"If we opened the fridge, there wouldn't be any, you know, 'oh well, I didn't even pick it up.'"

Lennon thought Beett was scared

Context 5:
It was  my final performance in marching band. The band took me around for 6 months, covering damaged units and buildings since I was 13 years old. After coming up from the army, I mastered it and was wild with emotion, it all began

Context 6:
Jim found an old disposable camera in the bottom of his junk drawer. He thought it was worth a mint.

Traditionally, young men spending their careers with their wives, girlfriends, and children must up their financial measures frequently and securely. This is primary among

Context 7:
Ron started his new job as a landscaper today.

He's happy. It's nice.

Rep. Chris Van Hollen, R-Md., talks about his plans to renew the agency's existing Conservation Plan for the Vieaux

Context 8:
John and Billy became very skilled at beer pong. She broke down the shipping line and started cooking over again. It wasn't changed once she got to Huntington to begin trading.

Monson

He was the genius of the trade, and

Context 9:
Caroline was a student in medical school.

A luxury car that had become a more practical passenger is still parked just off Sacramento Street, waiting for a spot in a long line of residents on a path that was recently rebuilt.



Context 10:
Trish hated the outdoors. They'd played tables at wedding parties for both Barack and Mike. Captain James Cameron of Lost, for god's sake, had that distaste and Phil Russell of Friends made him the first witness, but
```

4. Free response about ancestral sampling: 

Ancesteral sampling is a good alternative to greedy search. It solves the looping problem by 'drawing' a next word based on the likelihood of that word being next instead of always picking the most probable one. We can see this as none of the prompt continuations have any loops. What this program struggles with is being too random. This is apparent as in this output very low probability words such as Grifled and Gedaufar Glennist are produced by the model. These words are very clearly at the bottom of the probability distribution and highlight ancesterals sampling of allowing *all* potential words to be considered. Because ancesteral sampling doesn't just pick the most probable word like greedy search does, it has a lot more creative responses. It's drawback though is that its responses are too random and become incoherent. This leads us into the top-k and top-p methods which create a middle ground between these two.

## Part 2

6. Top-k decoding outputs:
```
Context 1:
Rick grew up in a troubled household. He went to the local school and attended the Eastern Washington Jesuit School where John E. Bauers taught English. When he began attending the university, he was a regular on the campus radio station and

Context 2:
Laverne needs to prepare something for her friend's party. "You have someone who can put someone in the right place and it's good for them," she says.

The first thing she will do is take out her kids and start the party first

Context 3:
Sarah had been dreaming of visiting Europe for years. "The first time she visited the world was in 1989," a family friend said, when she arrived. "She was so passionate about trying to get back in Europe that she brought me a copy with

Context 4:
Gina was worried the cookie dough in the tube would be gross. So she started baking the peanut butter. "We did that on top of the bread, and it tasted great, too." She says that she baked it in 1 1/2 pounds or 7 1

Context 5:
It was  my final performance in marching band.  I am not going to describe it as "good music."  It was  something more like "music of some sort."  I'm not a drummer, I'm not

Context 6:
Jim found an old disposable camera in the bottom of his junk drawer. "What's the matter, Mr. Peek?" he said. "A man. The police don't like it, after all."


"What?" I asked again and again.



Context 7:
Ron started his new job as a landscaper today.

"I work for a couple years and now I'm on my last job. The good news is, it's not a big deal. And you know if they asked, 'Why don

Context 8:
John and Billy became very skilled at beer pong. I think in the last 15 years I've spent a lot of time at the Brewster's Beer Farm & Wine Farm which is a really good thing because they have had a lot of beer pong

Context 9:
Caroline was a student in medical school. She and her dad, Johnnie, used to get together at McDonald's with their five kids. While they were on vacation, Johnnie was taking a nap in the living room when two men,

Context 10:
Trish hated the outdoors. "Well you got in a lot of trouble," her father added.

But she always said: "You could say I was the one who really did it."

The mother and her
```

7. Free response about top-k decoding:

Top-k provides a good middle ground between greedy search and ancesteral sampling as doesn't just pick the most probable option, but also limits the tail of the probability distribution. We see this in the results as the responses are much more creative than the greedy search and do not get stuck in loops. At the same time, it is not as random as ancesteral sampling. The responses tend to be much more coherent than the ancesteral search output and do not output any crazy (but fun!) names like Gedaufar Glennist. Still, a lot of these responses drift off topic very quickly. I think this is due to the k value of 40 which seems high to me and probably leaves in a lot of low probability words in the distribution. This can be particularly problematic when there is a super high likelihood of one or two words following a string and top-k leaves 38 others in the distribution that don't really make sense. One example of this is in context 4 when the response says "she started baking the peanut butter." There are probably a handful very likely things the subject could be baking and peanut butter was probably very unlikely, but because there are so many words left in peanut butter got selected from a very low likelihood in the tail. This particular issue can be mitigated by top-p, but we will see if that has other issues in the next prompt...

8. Top-p decoding outputs:
```
Context 1:
Rick grew up in a troubled household. He has since moved to New Orleans, where he understands what it means to live in need and understands that this is not a place for personal growth and expansion.

Like Beth Meyers, Kai

Context 2:
Laverne needs to prepare something for her friend's party. She'll go to Grandma and bring home the big bomb. The mood strikes, now that he's in the middle of getting it. And how long is it going to be?

She

Context 3:
Sarah had been dreaming of visiting Europe for years. The guards kept his dreams in perspective. He was conscious of his isolation in a room full of children and mothers, toys sitting in his arms, and nightmares of aging—he was 15 years old when

Context 4:
Gina was worried the cookie dough in the tube would be gross. But she also learned that something had messed up. That a vendor would drive up to her mom's home with a tray of pot smoking toys. Just as she had planned, they brought them over to

Context 5:
It was  my final performance in marching band. It was my last post of the game before CPS. After the game, they decided to call me on VIR-45 and assume this was what I was doing in the game. So I

Context 6:
Jim found an old disposable camera in the bottom of his junk drawer. He wanted to try and buy a camera to take photos of his daughter going to her 2 p.m. special, but they wouldn't be able to be sold.

At age 5,

Context 7:
Ron started his new job as a landscaper today. He's very excited about the concept behind the reworking. "It's a visually appealing piece," he says, "and I think it has a lot of appeal because it's more accessible."


Context 8:
John and Billy became very skilled at beer pong. We sent no soldiers. Two years later we was finally sent to Afghanistan as part of the M1 Task Force, and to resume our British-Chinese mission in Iran at the Suez Canal."


Context 9:
Caroline was a student in medical school.

Read more about the investigation below:A upvote for Chris Stevens who lost in the redraw, was given every ounce of hope he had after yet another one by fans for a backup

Context 10:
Trish hated the outdoors. It was a touch unfair to most women, but that wasn't what scared her most, and that was when she was elected in May 2017 as the youngest woman in a Senate race in Louisiana.

```

9. Free response about top-p decoding:

Top-p is very similar to top-k in the idea that we don't want to always pick the most probable next word, but also want to limit the words far in the tail of the distribution from being selected. Instead of limiting by a certain number of candidate words, top-p limits but the top x amount of words that have a cumulative probability of y. Like top-k, this strikes a middle ground between greedy search and acesteral search. Unlike top-k, when there are one or two candidate words make up almost all of the probability (i.e. the distribution has heavy skew), top-p is able to cut out improbable words that top-k would keep in avoiding the issue discussed in the last free response. This works well for very specific prompts, but when we input more open ended, story like prompts as we did in this assignment, the responses get very random. While both top-k and top-p are pretty random in what they spit out, top-p gets off topic much quicker and tends to stay on very specific topics once its in it. A good example of this is the beer pong prompt where Billy and Johnny immediately become soldiers and go to Afghanistan. This highlights top-p's issues which is when probability distributions are very flat, there might be hundereds of words included in the probability distribution. Context 8 provides a really good example of this because after the prompt, there are probably tons of different things that the prompt could say, and while soldier probably was not in the top 40 words, it was in the top 90% of likelihood. Then, once the model gets into a very specific topic, such as soldiers and war, it sticks to it for the rest of the response as their are probably only a limited amount of highly probable words in the training that follow soldiers, Afghanistan, and task force. 

## Part 3

12. Values chosen for temperature and k:

    Tempurature: 0.75, 1, 1.25

    k values: 5, 30, 60
    
14. Features chosen for qualitative measurement:
These five features will be what the output is evaluated on and rate 1-5. The best cumulative score will be selected.
    1. Comprehention - Is the output a plausable sentence? Can I comprehend an idea from it?
    2. Grammar - Does the grammar and sentence structure make sense
    3. Creativity - How creative is the sentence?
    4. Theme - Does the output have a coherent idea or theme throughout the response?
    5. Naturalness - Does it sound like a real person wrote this?
15. Results:

|          | k = 5 | k = 30 | k = 60 |
|----------|-------|--------|--------|
| T = 0.75 | Comprehention: 3<br>Grammar: 3<br>Creativity: 2<br>Theme: 4<br>Naturalness: 2| Comprehention: 2<br>Grammar: 3<br>Creativity: 3<br>Theme: 3<br>Naturalness: 2| Comprehention: 1<br>Grammar: 2<br>Creativity: 3<br>Theme: 4<br>Naturalness: 1|
| T = 1.00 | Comprehention: 4<br>Grammar: 4<br>Creativity: 3<br>Theme: 4<br>Naturalness: 3| Comprehention: 3<br>Grammar: 3.5<br>Creativity: 4<br>Theme: 3<br>Naturalness: 3| Comprehention: 3<br>Grammar: 3<br>Creativity: 4<br>Theme: 2<br>Naturalness: 2|
| T = 1.25 | Comprehention: 4<br>Grammar: 4<br>Creativity: 3<br>Theme: 3<br>Naturalness: 3| Comprehention: 4<br>Grammar: 3<br>Creativity: 4<br>Theme: 3.5<br>Naturalness: 3| Comprehention: 4<br>Grammar: 3<br>Creativity: 4<br>Theme: 3<br>Naturalness: 3|


15. Outputs for best temperature x k combination: 

The best temperature x k combination was a temperature of 1 and a k of 5.
```
Context 1:
Rick grew up in a troubled household. He was a single father who worked at an auto repair shop. But he didn't have the same financial problems as his older brother. He had been diagnosed with cancer and was struggling to make ends meet

Context 2:
Laverne needs to prepare something for her friend's party. He is also a big fan of the show.

"I don't think she wants to get into it," he says. "She wants to get to know the guys. I don't

Context 3:
Sarah had been dreaming of visiting Europe for years.

She told the BBC: "I've never really felt like travelling. I think that's what I was thinking when I started.

"The first time I went to a museum in

Context 4:
Gina was worried the cookie dough in the tube would be gross. She said, "I just want to make sure it's not too gross. It's a little bit too soft. I don't know if that's a problem."

"I think we

Context 5:
It was  my final performance in marching band. The last day was on the 10th of May. The last day was the 20th of June. I went to the theatre with a band and I went to my room with a band.


Context 6:
Jim found an old disposable camera in the bottom of his junk drawer. He took it out to check it out, and it had an amazing view of his family and the people that were around them. It was a very nice camera, but it was very expensive.


Context 7:
Ron started his new job as a landscaper today. He said it's a good way to start a new life. He said he's happy with the new home he has in his home.

"I feel like we are going to have a

Context 8:
John and Billy became very skilled at beer pong. The two had a love of music and were very good friends, and they had a very good time. I had never seen Billy in my life and he was very friendly and very friendly with me.

Context 9:
Caroline was a student in medical school.

"I'm a little surprised that they took her, but I don't think it was an isolated case," said Dr. John P. Schmitt, director of the Center for the Study

Context 10:
Trish hated the outdoors. "I don't like to go outside in my life, but there are some people in my life who do," she told me recently. "They are the most dangerous people in the world. And
```
