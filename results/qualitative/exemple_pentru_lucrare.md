# Anexă: Exemple calitative pentru analiza erorilor
Generat din experimentul exp01a_full (DistilBERT, full fine-tuning)


## Action

### ✅ Predicție corectă (True Positive)

**Exemplu 1** (sample_id: t1_givys64)
> Germany has upset other EU member states by securing a disproportionately large share of the bloc’s common pool of vaccines, according to a report. 
 Brussels  has ordered  roughly two billion doses in total, which are theoretically to be divided up according to each member state’s population. Howev...
- **Gold:** `[12-17]` → "upset"
- **Predicție:** `[12-17]` → "upset"
- **IoU:** 1.00

**Exemplu 2** (sample_id: t1_cguw8w8)
> I saw a friend post this elsewhere as I was finishing my coffee and found it fascinating. The potency of PET's arguments are pretty remarkable. There has been a significant amount of mythologizing surrounding constitutional reform talks, and the events surrounding repatriation of the constitution.  ...
- **Gold:** `[105-110]` → "PET's"
- **Predicție:** `[105-120]` → "PET's arguments"
- **IoU:** 0.50

**Exemplu 3** (sample_id: t1_dzqz120)
> This video discusses the way companies are trying to change Bitcoin to Blockchain - a meaningless, bullshit term that corporations are using for marketing purposes. 
 This ties directly to the fake DASH/ Mondsanto article heavily guided to the top of our sub in an obvious Astroturf fashion.  
 I cal...
- **Gold:** `[53-81]` → "change Bitcoin to Blockchain"
- **Predicție:** `[39-81]` → "are trying to change Bitcoin to Blockchain"
- **IoU:** 0.57

### ❌ Predicție inventată (False Positive)

**Exemplu 1** (sample_id: t1_givys64)
> Germany has upset other EU member states by securing a disproportionately large share of the bloc’s common pool of vaccines, according to a report. 
 Brussels  has ordered  roughly two billion doses in total, which are theoretically to be divided up according to each member state’s population. Howev...
- **Predicție:** `[44-88]` → "securing a disproportionately large share of"
- **IoU:** 0.00

**Exemplu 2** (sample_id: t1_givys64)
> Germany has upset other EU member states by securing a disproportionately large share of the bloc’s common pool of vaccines, according to a report. 
 Brussels  has ordered  roughly two billion doses in total, which are theoretically to be divided up according to each member state’s population. Howev...
- **Predicție:** `[100-123]` → "common pool of vaccines"
- **IoU:** 0.00

**Exemplu 3** (sample_id: t1_givys64)
> Germany has upset other EU member states by securing a disproportionately large share of the bloc’s common pool of vaccines, according to a report. 
 Brussels  has ordered  roughly two billion doses in total, which are theoretically to be divided up according to each member state’s population. Howev...
- **Predicție:** `[164-208]` → "ordered  roughly two billion doses in total,"
- **IoU:** 0.00

### ⚠️ Span ratat complet (False Negative)

**Exemplu 1** (sample_id: t1_f7ju17o)
> A great article on what's taking place in Bolivia, referencing some similar US backed coups in the region as well as recounting some of Bolivia's history and western policy towards the country.
- **Gold:** `[86-91]` → "coups"
- **IoU:** 0.00

**Exemplu 2** (sample_id: t1_joq538t)
> Redditors are, just like most social media users, highly motivated to oppose things they see as evil. This motivation, coupled with moderators permitting ontological hatred of an entire group of people, is what gives Redditors an excuse to go on the offensive without pesky restrictions such as “empa...
- **Gold:** `[143-172]` → "permitting ontological hatred"
- **IoU:** 0.00

**Exemplu 3** (sample_id: t1_cguw8w8)
> I saw a friend post this elsewhere as I was finishing my coffee and found it fascinating. The potency of PET's arguments are pretty remarkable. There has been a significant amount of mythologizing surrounding constitutional reform talks, and the events surrounding repatriation of the constitution.  ...
- **Gold:** `[354-357]` → "(out of excerpt)"
- **IoU:** 0.00

### 🟡 Span aproape, IoU sub 0.5 (Partial)

**Exemplu 1** (sample_id: t1_hbkbe8c)
> I assume most people hear are familiar with the idea that an old world conspiracy that came to fruition in the Anglo monarchies and resulted in the colonization/control of most of the world. Their largely Egyptian themed monuments are everywhere in America. Does resistance to this conspiracy include...
- **Gold:** `[132-168]` → "resulted in the colonization/control"
- **Predicție:** `[144-171]` → "the colonization/control of"
- **IoU:** 0.40

**Exemplu 2** (sample_id: t1_k4fu291)
> Palestine supporters have been attempting cyber-attacks on Indian government websites, including those of the Delhi government and AIIMS, accusing India of siding with Israel, sources said Monday, setting off a retaliatory blitz. The Indian cyber establishment has successfully fended off attacks in ...
- **Gold:** `[31-86]` → "attempting cyber-attacks on Indian government websites,"
- **Predicție:** `[31-55]` → "attempting cyber-attacks"
- **IoU:** 0.33

**Exemplu 3** (sample_id: t1_haw5wob)
> NYC Mandates Vaccine in order to participate in "society". 
 My Body, My Rules, My choice.  Segregation is wrong. 
 Take back NYC and USA from forced experimental injections on the people!  
 End GMO Humans!
- **Gold:** `[0-58]` → "NYC Mandates Vaccine in order to participate in "society"."
- **Predicție:** `[4-23]` → "Mandates Vaccine in"
- **IoU:** 0.33

## Actor

### ✅ Predicție corectă (True Positive)

**Exemplu 1** (sample_id: t1_givys64)
> Germany has upset other EU member states by securing a disproportionately large share of the bloc’s common pool of vaccines, according to a report. 
 Brussels  has ordered  roughly two billion doses in total, which are theoretically to be divided up according to each member state’s population. Howev...
- **Gold:** `[0-7]` → "Germany"
- **Predicție:** `[0-7]` → "Germany"
- **IoU:** 1.00

**Exemplu 2** (sample_id: t1_joq538t)
> Redditors are, just like most social media users, highly motivated to oppose things they see as evil. This motivation, coupled with moderators permitting ontological hatred of an entire group of people, is what gives Redditors an excuse to go on the offensive without pesky restrictions such as “empa...
- **Gold:** `[0-9]` → "Redditors"
- **Predicție:** `[0-9]` → "Redditors"
- **IoU:** 1.00

**Exemplu 3** (sample_id: t1_joq538t)
> Redditors are, just like most social media users, highly motivated to oppose things they see as evil. This motivation, coupled with moderators permitting ontological hatred of an entire group of people, is what gives Redditors an excuse to go on the offensive without pesky restrictions such as “empa...
- **Gold:** `[217-226]` → "Redditors"
- **Predicție:** `[220-223]` → "dit"
- **IoU:** 1.00

### ❌ Predicție inventată (False Positive)

**Exemplu 1** (sample_id: t1_f7ju17o)
> A great article on what's taking place in Bolivia, referencing some similar US backed coups in the region as well as recounting some of Bolivia's history and western policy towards the country.
- **Predicție:** `[42-49]` → "Bolivia"
- **IoU:** 0.00

**Exemplu 2** (sample_id: t1_gv484ay)
> Apr 14, 2021 - Whether or not large bilateral trade imbalances are a signal of non-reciprocal (or ‘unfair’) trade costs has been the subject of debate for some time, and was brought to the fore during President Trump’s time in office. This column argues that if the trading partners’ average trade co...
- **Predicție:** `[201-218]` → "President Trump’s"
- **IoU:** 0.00

**Exemplu 3** (sample_id: t1_hbkbe8c)
> I assume most people hear are familiar with the idea that an old world conspiracy that came to fruition in the Anglo monarchies and resulted in the colonization/control of most of the world. Their largely Egyptian themed monuments are everywhere in America. Does resistance to this conspiracy include...
- **Predicție:** `[191-196]` → "Their"
- **IoU:** 0.00

### ⚠️ Span ratat complet (False Negative)

**Exemplu 1** (sample_id: t1_f7ju17o)
> A great article on what's taking place in Bolivia, referencing some similar US backed coups in the region as well as recounting some of Bolivia's history and western policy towards the country.
- **Gold:** `[76-78]` → "US"
- **IoU:** 0.00

**Exemplu 2** (sample_id: t1_givys64)
> Germany has upset other EU member states by securing a disproportionately large share of the bloc’s common pool of vaccines, according to a report. 
 Brussels  has ordered  roughly two billion doses in total, which are theoretically to be divided up according to each member state’s population. Howev...
- **Gold:** `[304-311]` → "(out of excerpt)"
- **IoU:** 0.00

**Exemplu 3** (sample_id: t1_joq538t)
> Redditors are, just like most social media users, highly motivated to oppose things they see as evil. This motivation, coupled with moderators permitting ontological hatred of an entire group of people, is what gives Redditors an excuse to go on the offensive without pesky restrictions such as “empa...
- **Gold:** `[132-142]` → "moderators"
- **IoU:** 0.00

### 🟡 Span aproape, IoU sub 0.5 (Partial)

**Exemplu 1** (sample_id: t1_ghbm81l)
> mins long. Corbett does a great job compiling all the information and presenting it all in an easy to follow video touching on the history of Technocracy, NWO agendas being rebranded and crammed in our faces as the (not so) Great Reset and their end goal of transhumanism "utopia"
- **Gold:** `[142-166]` → "Technocracy, NWO agendas"
- **Predicție:** `[148-150]` → "cr"
- **IoU:** 0.33

**Exemplu 2** (sample_id: t1_g3wequq)
> Bill Gates and Nathan Myhrvold are huge names when it comes to Microsoft. What were they doing with a sex trafficker? 
 Myhrvold, former Chief Technology Officer at Microsoft, and Epstein were pictured at the  Edge's  "Billionaires' Dinner" in 2000. He's also in his  Black Book frequently . 
 And wh...
- **Gold:** `[0-30]` → "Bill Gates and Nathan Myhrvold"
- **Predicție:** `[0-10]` → "Bill Gates"
- **IoU:** 0.40

**Exemplu 3** (sample_id: t1_g3wequq)
> Bill Gates and Nathan Myhrvold are huge names when it comes to Microsoft. What were they doing with a sex trafficker? 
 Myhrvold, former Chief Technology Officer at Microsoft, and Epstein were pictured at the  Edge's  "Billionaires' Dinner" in 2000. He's also in his  Black Book frequently . 
 And wh...
- **Gold:** `[120-187]` → "Myhrvold, former Chief Technology Officer at Microsoft, and Epstein"
- **Predicție:** `[180-187]` → "Epstein"
- **IoU:** 0.11

## Effect

### ✅ Predicție corectă (True Positive)

**Exemplu 1** (sample_id: t1_cguw8w8)
> I saw a friend post this elsewhere as I was finishing my coffee and found it fascinating. The potency of PET's arguments are pretty remarkable. There has been a significant amount of mythologizing surrounding constitutional reform talks, and the events surrounding repatriation of the constitution.  ...
- **Gold:** `[209-237]` → "constitutional reform talks,"
- **Predicție:** `[209-237]` → "constitutional reform talks,"
- **IoU:** 1.00

**Exemplu 2** (sample_id: t1_cguw8w8)
> I saw a friend post this elsewhere as I was finishing my coffee and found it fascinating. The potency of PET's arguments are pretty remarkable. There has been a significant amount of mythologizing surrounding constitutional reform talks, and the events surrounding repatriation of the constitution.  ...
- **Gold:** `[246-298]` → "events surrounding repatriation of the constitution."
- **Predicție:** `[246-298]` → "events surrounding repatriation of the constitution."
- **IoU:** 1.00

**Exemplu 3** (sample_id: t1_ih5cggk)
> An unvaccinated young man in Rockland County, NY has contracted paralytic polio that is consistent with the source from a live attenuated vaccine. It was contracted in the US as there was no international travel. It seems that they must have contracted the virus from someone who had received the liv...
- **Gold:** `[53-79]` → "contracted paralytic polio"
- **Predicție:** `[53-79]` → "contracted paralytic polio"
- **IoU:** 1.00

### ❌ Predicție inventată (False Positive)

**Exemplu 1** (sample_id: t1_joq538t)
> Redditors are, just like most social media users, highly motivated to oppose things they see as evil. This motivation, coupled with moderators permitting ontological hatred of an entire group of people, is what gives Redditors an excuse to go on the offensive without pesky restrictions such as “empa...
- **Predicție:** `[70-83]` → "oppose things"
- **IoU:** 0.00

**Exemplu 2** (sample_id: t1_joq538t)
> Redditors are, just like most social media users, highly motivated to oppose things they see as evil. This motivation, coupled with moderators permitting ontological hatred of an entire group of people, is what gives Redditors an excuse to go on the offensive without pesky restrictions such as “empa...
- **Predicție:** `[96-100]` → "evil"
- **IoU:** 0.00

**Exemplu 3** (sample_id: t1_cguw8w8)
> I saw a friend post this elsewhere as I was finishing my coffee and found it fascinating. The potency of PET's arguments are pretty remarkable. There has been a significant amount of mythologizing surrounding constitutional reform talks, and the events surrounding repatriation of the constitution.  ...
- **Predicție:** `[187-190]` → "olo"
- **IoU:** 0.00

### ⚠️ Span ratat complet (False Negative)

**Exemplu 1** (sample_id: t1_givys64)
> Germany has upset other EU member states by securing a disproportionately large share of the bloc’s common pool of vaccines, according to a report. 
 Brussels  has ordered  roughly two billion doses in total, which are theoretically to be divided up according to each member state’s population. Howev...
- **Gold:** `[55-73]` → "disproportionately"
- **IoU:** 0.00

**Exemplu 2** (sample_id: t1_cguw8w8)
> I saw a friend post this elsewhere as I was finishing my coffee and found it fascinating. The potency of PET's arguments are pretty remarkable. There has been a significant amount of mythologizing surrounding constitutional reform talks, and the events surrounding repatriation of the constitution.  ...
- **Gold:** `[111-120]` → "arguments"
- **IoU:** 0.00

**Exemplu 3** (sample_id: t1_gzs5v7g)
> On July 16, 1996, 36-year-old Susan Walsh left her Nutley, New Jersey apartment to use a pay phone across the street – a routine habit since she didn’t have a home telephone. She was never seen again. 
20 years later, the disappearance of Susan Walsh continues to be one of the more intriguing missin...
- **Gold:** `[42-46]` → "left"
- **IoU:** 0.00

### 🟡 Span aproape, IoU sub 0.5 (Partial)

**Exemplu 1** (sample_id: t1_jolxlod)
> This is shocking. Listen to the video.  
 This IDF soldier (who rather stupidly disclosed his Battalion and Brigade and Unit), discloses that they are homophobic, racist and particularly religious.  
 He also disclosed that an  80  year old man was arrested and messed around with and locked up overn...
- **Gold:** `[332-352]` → "(out of excerpt)"
- **Predicție:** `[347-352]` → "(out of excerpt)"
- **IoU:** 0.20

**Exemplu 2** (sample_id: t1_ectb45o)
> This is seriously messed up. The rich are literally taking children's blood and using it to reverse the aging process. Experiments have shown positive effects in mice. 
 "So it does appear to turn back the clock on aging in humans," says Dr.  Karmazin . "People feel and look younger after just one t...
- **Gold:** `[344-442]` → "(out of excerpt)"
- **Predicție:** `[344-352]` → "(out of excerpt)"
- **IoU:** 0.07

**Exemplu 3** (sample_id: t1_k7maqpi)
> Article highlights: 
 1: 
 
 A candidate 'short list' to replace Dr. Kirkpatrick at the All-Domain Anomaly Resolution Office (AARO) has been interviewed, an ex-official told DailyMail.com 
 
 2: 
 
 The Pentagon's UFO chief will resign by year's end — amid a wave of complaints accusing him of making...
- **Gold:** `[259-277]` → "wave of complaints"
- **Predicție:** `[267-277]` → "complaints"
- **IoU:** 0.33

## Evidence

### ✅ Predicție corectă (True Positive)

**Exemplu 1** (sample_id: t1_f7ju17o)
> A great article on what's taking place in Bolivia, referencing some similar US backed coups in the region as well as recounting some of Bolivia's history and western policy towards the country.
- **Gold:** `[8-15]` → "article"
- **Predicție:** `[2-15]` → "great article"
- **IoU:** 0.50

**Exemplu 2** (sample_id: t1_hsoj4b0)
> These surveys paint an extremely scary picture on the authoritarianism creep of the left when it comes to Americans who question or refuse to get the vaccine--for whatever reason (e.g., natural immunity). Everything from fines, incarceration, to the placement of unvaccinated into "designated facilit...
- **Gold:** `[0-13]` → "These surveys"
- **Predicție:** `[0-13]` → "These surveys"
- **IoU:** 1.00

**Exemplu 3** (sample_id: t1_ih5cggk)
> An unvaccinated young man in Rockland County, NY has contracted paralytic polio that is consistent with the source from a live attenuated vaccine. It was contracted in the US as there was no international travel. It seems that they must have contracted the virus from someone who had received the liv...
- **Gold:** `[88-146]` → "consistent with the source from a live attenuated vaccine."
- **Predicție:** `[85-146]` → "is consistent with the source from a live attenuated vaccine."
- **IoU:** 0.90

### ❌ Predicție inventată (False Positive)

**Exemplu 1** (sample_id: t1_givys64)
> Germany has upset other EU member states by securing a disproportionately large share of the bloc’s common pool of vaccines, according to a report. 
 Brussels  has ordered  roughly two billion doses in total, which are theoretically to be divided up according to each member state’s population. Howev...
- **Predicție:** `[135-147]` → "to a report."
- **IoU:** 0.00

**Exemplu 2** (sample_id: t1_givys64)
> Germany has upset other EU member states by securing a disproportionately large share of the bloc’s common pool of vaccines, according to a report. 
 Brussels  has ordered  roughly two billion doses in total, which are theoretically to be divided up according to each member state’s population. Howev...
- **Predicție:** `[411-461]` → "(out of excerpt)"
- **IoU:** 0.00

**Exemplu 3** (sample_id: t1_cguw8w8)
> I saw a friend post this elsewhere as I was finishing my coffee and found it fascinating. The potency of PET's arguments are pretty remarkable. There has been a significant amount of mythologizing surrounding constitutional reform talks, and the events surrounding repatriation of the constitution.  ...
- **Predicție:** `[6-34]` → "a friend post this elsewhere"
- **IoU:** 0.00

### ⚠️ Span ratat complet (False Negative)

**Exemplu 1** (sample_id: t1_gzs5v7g)
> On July 16, 1996, 36-year-old Susan Walsh left her Nutley, New Jersey apartment to use a pay phone across the street – a routine habit since she didn’t have a home telephone. She was never seen again. 
20 years later, the disappearance of Susan Walsh continues to be one of the more intriguing missin...
- **Gold:** `[145-174]` → "didn’t have a home telephone."
- **IoU:** 0.00

**Exemplu 2** (sample_id: t1_gzs5v7g)
> On July 16, 1996, 36-year-old Susan Walsh left her Nutley, New Jersey apartment to use a pay phone across the street – a routine habit since she didn’t have a home telephone. She was never seen again. 
20 years later, the disappearance of Susan Walsh continues to be one of the more intriguing missin...
- **Gold:** `[283-342]` → "(out of excerpt)"
- **IoU:** 0.00

**Exemplu 3** (sample_id: t1_eoryiia)
> This article is being pushed on my "Front page of Google" It attempts to tie Gabbard to conspiracy theories and discredit both. Also noticed all the recommended articles at the bottom are similarly titled and quite misrepresenting of Gabbard.
- **Gold:** `[50-57]` → "Google""
- **IoU:** 0.00

### 🟡 Span aproape, IoU sub 0.5 (Partial)

**Exemplu 1** (sample_id: t1_givys64)
> Germany has upset other EU member states by securing a disproportionately large share of the bloc’s common pool of vaccines, according to a report. 
 Brussels  has ordered  roughly two billion doses in total, which are theoretically to be divided up according to each member state’s population. Howev...
- **Gold:** `[138-139]` → "a"
- **Predicție:** `[135-147]` → "to a report."
- **IoU:** 0.33

**Exemplu 2** (sample_id: t1_givys64)
> Germany has upset other EU member states by securing a disproportionately large share of the bloc’s common pool of vaccines, according to a report. 
 Brussels  has ordered  roughly two billion doses in total, which are theoretically to be divided up according to each member state’s population. Howev...
- **Gold:** `[414-416]` → "(out of excerpt)"
- **Predicție:** `[411-461]` → "(out of excerpt)"
- **IoU:** 0.12

**Exemplu 3** (sample_id: t1_cguw8w8)
> I saw a friend post this elsewhere as I was finishing my coffee and found it fascinating. The potency of PET's arguments are pretty remarkable. There has been a significant amount of mythologizing surrounding constitutional reform talks, and the events surrounding repatriation of the constitution.  ...
- **Gold:** `[8-19]` → "friend post"
- **Predicție:** `[6-34]` → "a friend post this elsewhere"
- **IoU:** 0.40

## Victim

### ✅ Predicție corectă (True Positive)

**Exemplu 1** (sample_id: t1_hsoj4b0)
> These surveys paint an extremely scary picture on the authoritarianism creep of the left when it comes to Americans who question or refuse to get the vaccine--for whatever reason (e.g., natural immunity). Everything from fines, incarceration, to the placement of unvaccinated into "designated facilit...
- **Gold:** `[263-275]` → "unvaccinated"
- **Predicție:** `[265-272]` → "vaccina"
- **IoU:** 1.00

**Exemplu 2** (sample_id: t1_eoryiia)
> This article is being pushed on my "Front page of Google" It attempts to tie Gabbard to conspiracy theories and discredit both. Also noticed all the recommended articles at the bottom are similarly titled and quite misrepresenting of Gabbard.
- **Gold:** `[234-242]` → "Gabbard."
- **Predicție:** `[231-242]` → "of Gabbard."
- **IoU:** 0.50

**Exemplu 3** (sample_id: t1_ih5cggk)
> An unvaccinated young man in Rockland County, NY has contracted paralytic polio that is consistent with the source from a live attenuated vaccine. It was contracted in the US as there was no international travel. It seems that they must have contracted the virus from someone who had received the liv...
- **Gold:** `[435-439]` → "(out of excerpt)"
- **Predicție:** `[435-438]` → "(out of excerpt)"
- **IoU:** 1.00

### ❌ Predicție inventată (False Positive)

**Exemplu 1** (sample_id: t1_joq538t)
> Redditors are, just like most social media users, highly motivated to oppose things they see as evil. This motivation, coupled with moderators permitting ontological hatred of an entire group of people, is what gives Redditors an excuse to go on the offensive without pesky restrictions such as “empa...
- **Predicție:** `[195-202]` → "people,"
- **IoU:** 0.00

**Exemplu 2** (sample_id: t1_hsoj4b0)
> These surveys paint an extremely scary picture on the authoritarianism creep of the left when it comes to Americans who question or refuse to get the vaccine--for whatever reason (e.g., natural immunity). Everything from fines, incarceration, to the placement of unvaccinated into "designated facilit...
- **Predicție:** `[106-115]` → "Americans"
- **IoU:** 0.00

**Exemplu 3** (sample_id: t1_gzs5v7g)
> On July 16, 1996, 36-year-old Susan Walsh left her Nutley, New Jersey apartment to use a pay phone across the street – a routine habit since she didn’t have a home telephone. She was never seen again. 
20 years later, the disappearance of Susan Walsh continues to be one of the more intriguing missin...
- **Predicție:** `[30-41]` → "Susan Walsh"
- **IoU:** 0.00

### ⚠️ Span ratat complet (False Negative)

**Exemplu 1** (sample_id: t1_f7ju17o)
> A great article on what's taking place in Bolivia, referencing some similar US backed coups in the region as well as recounting some of Bolivia's history and western policy towards the country.
- **Gold:** `[42-50]` → "Bolivia,"
- **IoU:** 0.00

**Exemplu 2** (sample_id: t1_f7ju17o)
> A great article on what's taking place in Bolivia, referencing some similar US backed coups in the region as well as recounting some of Bolivia's history and western policy towards the country.
- **Gold:** `[136-145]` → "Bolivia's"
- **IoU:** 0.00

**Exemplu 3** (sample_id: t1_givys64)
> Germany has upset other EU member states by securing a disproportionately large share of the bloc’s common pool of vaccines, according to a report. 
 Brussels  has ordered  roughly two billion doses in total, which are theoretically to be divided up according to each member state’s population. Howev...
- **Gold:** `[18-23]` → "other"
- **IoU:** 0.00

### 🟡 Span aproape, IoU sub 0.5 (Partial)

**Exemplu 1** (sample_id: t1_duozdp9)
> An estimated 77 million Americans have a debt that has been transferred to a private collection agency. Thousands have ended up in jail over debts as small as $28, with African-Americans and Hispanics the most affected. 
 The findings come from a new report by the American Civil Liberties Union (ACL...
- **Gold:** `[493-630]` → "(out of excerpt)"
- **Predicție:** `[493-507]` → "(out of excerpt)"
- **IoU:** 0.08

**Exemplu 2** (sample_id: t1_ea4qpej)
> This is a very interesting turn of events. Mexicans do not believe in putting Hondurans before their own citizens, but Americans, Germans, Brits and Swedes must do this or they are racist. 
 Maybe the absurd tide is turning?
- **Gold:** `[119-144]` → "Americans, Germans, Brits"
- **Predicție:** `[139-143]` → "Brit"
- **IoU:** 0.33

**Exemplu 3** (sample_id: t1_hwpsf4k)
> it’s no secret that Dr Hal Puthoff’s career has been “stigmatized” as even he calls it, allegedly due to the fact that extrasensory-perception, parapsychology, telekinesis, remote viewing & so much more of such “superhuman techniques” are indeed possible & real. This episode delves into the differen...
- **Gold:** `[20-36]` → "Dr Hal Puthoff’s"
- **Predicție:** `[27-36]` → "Puthoff’s"
- **IoU:** 0.33
