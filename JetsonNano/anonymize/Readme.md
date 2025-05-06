## Embedded system on Jetson Nano

Trumpas projekto aprašymas ir struktūra:

* **Tikslas**: Realiojo laiko vaizdo nuasmeninimo sistema, veido aptikimas, skeleto aptikimas ir vaizdo transliacija per MediaMTX.
* **Struktūra**:

  * `anonymize/` – modulis veido aptikimui ir nuasmeninimui (ResNet10 SSD + OpenPose).
  * `mediamtx/` – MediaMTX konteinerio konfigūracija srautui valdyti.
  * `old/` – archyvinės, ankstesnės implementacijos skriptai.
  * `web/` – Flask pagrindu sukurta WEB UI su HLS transliacija ir API.

---

## `anonymize/` modulis

Modulis atsakingas už veido aptikimą ir nuasmeninimą realiuoju laiku.

* **Failų struktūra**:

  * `run.py` – pagrindinis paleidimo skriptas, skaito vaizdo srautą, aptinka veidus ir perduoda tolimesniam apdorojimui.
  * `val.py` – vertina modelio tikslumą su COCO duomenimis.
  * `models/`, `modules/`, `datasets/` – modelių, pagalbinių modulių ir duomenų apdorojimo komponentai.
  * `deploy.prototxt` ir `res10_300x300_ssd_iter_140000.caffemodel` – veido aptikimo modelio konfigūracija ir svoriai.
  * `checkpoint_iter_370000.pth` – OpenPose skeleto modelio svoriai.

**Paleidimas**:

```bash
cd anonymize
python3 run.py --gst-in <įvesties-pipeline> --gst-out <išvesties-pipeline>
```

## `mediamtx/` konteineris

MediaMTX konteineris skirtas realaus laiko HLS/RTSP srautams valdyti.

* `docker-compose.yml` – Docker Compose aprašymas.
* `mediamtx.yml` – MediaMTX konfigūracija su įėjimų ir išėjimų nustatymais.

## `old/` katalogas

Seni, testiniai ir ankstesnių versijų skriptai, skirti skeleto ir veido nuasmeninimui.

---

## `web/` aplikacija 

Flask pagrindu sukurta valdymo sąsaja realaus laiko srautams ir sistemų metrikoms.

**Struktūra**:

* `static/` – CSS, JS ir medijos failai.
* `templates/` – HTML šablonai.
* `users.db` – SQLite naudotojų autentifikacijai.

---
