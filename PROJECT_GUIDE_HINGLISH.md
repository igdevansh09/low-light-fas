# Low-Light Face Anti-Spoofing Project: Simple Guide

## Ye project hai kya?

Ye Python app do face images compare karta hai: ek **reference image** (jis person ko pehle se pehchanna hai) aur ek **target image** (jise verify karna hai). Compare karne se pehle app check karta hai ki target face asli/live lag raha hai ya photo/screen jaisa spoof ho sakta hai.

Seedhi baat: app pehle poochta hai **"kya target face live lag raha hai?"**, phir **"kya ye reference wale person se match karta hai?"**

App browser mein Streamlit interface ke through chalta hai, lekin processing Python aur local models se hoti hai. Abhi input uploaded still images hain; ye live camera/video app nahi hai.

## App kaise kaam karta hai?

1. **Reference image upload karo.** Ye pehchaan ke liye trusted sample hai. Clear face wali image se achha result milne ki possibility hoti hai.
2. **Target image upload karo.** Ye woh image hai jisko check karna hai.
3. **Anti-spoofing check hota hai.** App target mein face region dhoondh kar do MiniFASNet models ko chalata hai. Dono ke predictions combine hote hain. Code mein class `1` ko live-face class maana gaya hai; baaki classes ko possible presentation attack ke roop mein reject kiya jata hai.
4. **Face detection hoti hai.** Agar liveness check pass ho, UniFace/RetinaFace reference aur target mein face aur landmarks dhoondhta hai.
5. **Identity compare hoti hai.** ArcFace dono faces ke numerical representations, yaani embeddings, banata hai. App un embeddings ki cosine similarity compare karta hai.
6. **Final result dikhta hai.** Similarity `0.40` ya usse zyada ho to app `IDENTITY VERIFIED` dikhata hai; isse kam ho to `IDENTITY MISMATCH`.

## Result ka matlab

- **Face Not Detected:** Anti-spoofing step ko usable face region nahi mila.
- **Presentation Attack:** Target ko live face class nahi mila, isliye identity matching aage nahi hoti.
- **Face Alignment Failed:** Reference ya target ke liye recognition wala detector usable face/landmarks nahi dhoondh paya.
- **Embedding Extraction Failed:** Face detect hua, par identity representation nahi ban saki.
- **Identity Verified:** Liveness pass hui aur similarity current `0.40` threshold tak pahunchi.
- **Identity Mismatch:** Liveness pass hui, lekin similarity threshold se kam rahi.

Screen par liveness score aur similarity score bhi dikhte hain. Ye debugging aur comparison ke liye useful hain, lekin inhe independently validated probability ya guarantee nahi samajhna chahiye.

## Andar models kya karte hain?

Is project mein alag models ke alag jobs hain. Face dhoondhna, spoof check karna, aur identity compare karna ek hi model ka kaam nahi hai.

### 1. Anti-spoof face detector

`antispoof_model/resources/detection_model/` mein OpenCV DNN ke through load hone wala RetinaFace/WiderFace detector hai. Ye image mein face ka bounding box dhoondhta hai; ye khud liveness ka final decision nahi karta. App is box se face crop banata hai aur MiniFASNet ke expected `80 x 80` input size ke hisaab se resize karta hai.

Identity-recognition side par UniFace ka alag RetinaFace detector use hota hai. Wo reference aur target dono mein face aur landmarks locate karta hai. Dono detector ka role face location hai, lekin anti-spoof crop aur recognition alignment alag pipeline steps hain.

### 2. MiniFASNet: live face ya spoof?

MiniFASNet ek chhota CNN (Convolutional Neural Network) hai. Image ko chhote learned filters se process karke uske visual patterns nikalta hai. Is model mein depthwise-separable blocks hain: depthwise convolution har feature channel ke andar spatial patterns dekhta hai, aur pointwise `1 x 1` convolution channels ko combine karta hai. Isse regular convolution ke muqable computation kam rakhne mein madad milti hai. Residual connections purane features ko aage ke layers tak add karti hain. `V1SE` model mein SE (Squeeze-and-Excitation) blocks channel importance ke hisaab se features ko weight bhi karte hain.

Model aakhir mein 3 class ke logits deta hai. Softmax un logits ko class scores/probabilities mein badalta hai:

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

App dono `.pth` models ke score vectors ko add karta hai. Jis class ka combined score sabse bada ho, us class ko final label banata hai. Current code mein **class `1` ko live** maana gaya hai; baaki labels reject hote hain. Repo mein baaki classes ke exact attack-type names ki mapping nahi hai, isliye unhe print/replay jaise specific attack labels kehna theek nahi hoga.

Displayed liveness score selected class ke combined score ko `2` se divide karta hai, kyunki current setup mein do weights hain. Ye dono models ka averaged score hai, calibrated real-world probability ya benchmark result nahi. Agar model weights ki count badle, to is hard-coded division ko bhi review karna hoga.

### 3. ArcFace: face ko number mein badalna

UniFace ke ArcFace recognizer ko detected face aur landmarks diye jaate hain. Wo face ka fixed-length numerical vector, yaani **embedding**, banata hai. Same identity ke embeddings aam taur par vector space mein closer hone chahiye; different identities ke embeddings farther apart. ArcFace naam angular-margin based face-recognition training approach ko refer karta hai, lekin is app mein ArcFace ko sirf pretrained recognizer ke roop mein call kiya gaya hai; is repository mein uska training code ya model weights maujood nahi hain.

App normalized reference vector $r$ aur target vector $t$ ke beech cosine similarity calculate karta hai:

$$\operatorname{cos\_sim}(r,t) = \frac{r \cdot t}{\|r\|_2\,\|t\|_2}$$

Score current app setting `0.40` ya zyada ho to match, warna mismatch. Ye threshold configurable decision rule hai, apne aap mein accuracy ya scientifically validated boundary nahi.

## Training code aur app inference ka difference

Repository mein anti-spoof model train karne ka code bhi hai, lekin wo Streamlit app ke har verification par nahi chalta. App seedhe do bundled MiniFASNet weights load karke inference karta hai. Training scripts mein `MultiFTNet` use hota hai: uska MiniFASNetV2SE backbone class logits banata hai, aur training-only `FTGenerator` Fourier-feature target ko predict karne ki koshish karta hai.

Dataset helper image ko grayscale karta hai, 2D Fast Fourier Transform nikalta hai, magnitude par `log(1 + |F|)` lagata hai, phir result normalize karta hai. Ye Fourier image training ke auxiliary target ke roop mein use hoti hai; current Streamlit inference path mein ise alag low-light enhancement ki tarah apply nahi kiya jata.

Training objective code mein classification cross-entropy aur Fourier map ka mean-squared error (MSE) mila kar banta hai:

$$L_{total} = 0.5\,L_{CE} + 0.5\,L_{MSE}$$

Cross-entropy sahi class ko high score dene ke liye model ko train karti hai. MSE predicted Fourier map aur generated target map ka average squared difference naapta hai. Config mein SGD optimizer, learning rate `0.1`, 25 epochs, batch size `1024`, aur learning-rate milestones diye hain. Ye configured values hain; inhe is baat ka proof na samjhein ki app ke bundled weights isi config ya isi dataset se train hue.

## Benchmark: abhi kis par result decide kiya gaya hai?

**Current honest answer: repository mein koi documented benchmark dataset, completed test protocol, ya numerical benchmark result nahi hai.** App ka `0.40` identity threshold code mein set hai; project files nahi dikhati ki ise held-out benchmark par tune ya validate kiya gaya. Bundled MiniFASNet weights ka original training dataset bhi yahan identify nahi hai. Training config `datasets/rgb_image/<patch_info>` folder expect karta hai, par ye path dataset ka naam nahi batata.

`antispoof_model/test.py` sample-image inference aur processing time dikhata hai. Wo labeled, held-out evaluation nahi karta aur APCER/BPCER/ACER calculate nahi karta; isliye use benchmark score nahi kehna chahiye. Progress report mein bhi benchmark results pending bataye gaye hain.

### Benchmark karna ho to kaise karein?

1. **Data ko define karo:** Bona fide/live faces, alag identities ke impostor pairs, aur presentation attacks (jaise print ya screen replay) collect/choose karo. Normal, dim aur very dim lighting alag record karo; dim level ko lux meter se measure karna behtar hai.
2. **Data split rakho:** Model/threshold decisions ke liye development set rakho. Final report ke liye identities aur, jahan mumkin ho, capture sessions ko alag rakh kar untouched test set banao. Test results dekh kar threshold dobara tune na karo.
3. **Liveness metrics nikalo:** Attack samples mein kitne galti se bona fide accept hue, uska APCER; genuine samples mein kitne galti se attack reject hue, uska BPCER. ACER in dono ka average hai:

	$$\operatorname{ACER} = \frac{\operatorname{APCER} + \operatorname{BPCER}}{2}$$

4. **Identity metrics nikalo:** Different people ko galti se same kehne ki rate FAR hai; same person ko galti se reject karne ki rate FRR hai. Threshold ko development set par badal kar FAR/FRR curve dekho. Jahan FAR aur FRR barabar hon, us operating point ko EER kehte hain; final test set par uski performance report karo.
5. **Conditions alag report karo:** Overall numbers ke saath normal/dim/very dim results, face-detection failures, embedding failures, liveness-score behavior, aur processing time alag note karo. Current threshold `0.40` ko baseline setting ke roop mein report karo; comparison ke liye development data par calibrated threshold bhi evaluate kiya ja sakta hai.

Is tarah decide hoga ki system kis condition mein kitna kaam karta hai. Sirf kuch demo images par `IDENTITY VERIFIED` aana low-light benchmark ya accuracy proof nahi hota.

## Kaun si technologies use hui hain?

- **Python:** Main programming language.
- **Streamlit:** Uploads aur results wala web interface.
- **OpenCV:** Uploaded images ko read karna aur anti-spoofing ke liye image crop karna.
- **PyTorch + MiniFASNet:** Target photo/screen spoof ho sakta hai ya live face, iska model-based check.
- **UniFace + RetinaFace:** Face aur facial landmarks dhoondhna.
- **ArcFace:** Face ko embedding mein badalna, jise compare kiya ja sake.
- **Cosine similarity:** Reference aur target embeddings kitne similar hain, uska score.

## Project ke important folders/files

- `app.py` — Main Streamlit app aur poora verification flow.
- `requirements.txt` — App chalane ke liye Python packages.
- `packages.txt` — Kuch Linux/system packages; hosted environment setup mein kaam aa sakte hain.
- `antispoof_model/resources/anti_spoof_models/` — App mein use hone wale do MiniFASNet model weights.
- `antispoof_model/resources/detection_model/` — Anti-spoofing crop ke liye face detector ki files.
- `antispoof_model/src/anti_spoof_predict.py` — Anti-spoofing model load aur inference.
- `antispoof_model/src/generate_patches.py` — Model ke liye face crop/patch banana.
- `antispoof_model/src/model_lib/` — MiniFASNet aur training ke model definitions.
- `antispoof_model/src/data_io/` — Training data load aur transform karne wale helpers.
- `antispoof_model/src/train_main.py`, `train.py`, `test.py` — Anti-spoofing model training/testing ka code.
- `project_synopsis.md`, `progress_report.md` — Project ka formal synopsis aur progress details.

## Windows par app kaise chalayein?

Terminal ko project ke main folder mein open karke ye commands chalao:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit terminal mein local URL dikhayega; use browser mein kholo. App use karne ke liye sidebar mein reference image aur main screen par target image upload karo. Models first run par load honge, isliye pehli baar thoda wait ho sakta hai.

Agar PowerShell virtual environment activate karne se mana kare, current terminal ke liye ye command chala kar activation phir try kar sakte ho:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

App ko project ke main folder se run karna important hai, kyunki model files ke paths isi folder structure ke hisaab se diye gaye hain.

## Low-light ke baare mein zaroori baat

Project ka goal low-light conditions mein face verification ko study karna hai. Lekin current app mein koi dedicated brightness enhancement step nahi hai, aur project files mein controlled low-light benchmark ya measured accuracy results nahi diye gaye hain. Isliye abhi ye kehna sahi nahi hoga ki system low light mein reliably kaam karta hai. Ye working prototype hai; low, normal aur bright lighting mein proper testing abhi karni baaki hai.

Similarity threshold `0.40` bhi app ka current fixed setting hai, test data se scientifically calibrate kiya hua value nahi. Isi tarah, screen par dikhne wala liveness score validation ke bina guaranteed probability nahi hai. Real biometric use se pehle privacy, consent, security, aur model performance ka proper evaluation zaroori hoga.

## Ek line mein summary

**Reference face + target face → pehle spoof/liveness check → phir face embedding comparison → verified, mismatch, ya rejection result.**
