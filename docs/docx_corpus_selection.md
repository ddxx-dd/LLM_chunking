# docx-corpus 문서 50개 수집 (vectara PDF 규모·구조에 맞춤, 2026-09-30)

`superdoc-dev/docx-corpus`에서 vectara PDF 50개와 글자 수가 비슷하고, 표가 적당하며, Word 스타일(제목)이 깔끔해서 Docling 변환 결과가 스타일별로 잘 나오는 docx 50개를 골랐다. **파이프라인은 건드리지 않았다.**

## 1. 결과 요약

- 최종 **50개 = 영어 47 + 한국어 3**. 한국어가 3개뿐인 것은 데이터셋의 한계다(아래 3절).
- 저장 위치: `data/docx_corpus/selected/{ko,en}/`(docx + Docling 변환 md 쌍), 목록·지표는 `data/docx_corpus/selected/selected.json`.
- 이전에 받은 50개(16개 선별)는 후보 풀(`data/docx_corpus/pool/`)로 옮겼다(삭제하지 않음). 후보 풀은 총 934개.

## 2. 선별 과정 (실측)

**vectara PDF 기준**: 변환 md 50개의 글자 수 중앙값 58,359자(10~90분위 24,982~101,048자), 제목 줄 수 중앙값 17.5개, 표 중앙값 1.0개(0~15).

**후보 풀**: 영어는 유형 7종(reports/policies/administrative/educational/technical/reference/legal)에서 유형당 100개 무작위(seed 42, 단어 수 4,000~18,000, confidence ≥0.7) 737개. 한국어는 단어 수 1,000~20,000인 문서 전부 197개.

| 단계 | ko | en |
|---|---|---|
| 후보 풀(다운로드 성공) | 197 | 737 |
| 1단계 통과(python-docx 지표) | 4 | 82 |
| 2단계+중복 제거 통과(Docling 변환 확인) | 3 | 56 |
| 최종(유형·계열 균형) | 3 | 47 |

**1단계 기준(python-docx, 변환 전)**: 총 글자 수 en 20,000~120,000 / ko 10,000~120,000(한국어는 글자당 정보량이 커서 하한을 낮춤); 제목 5개 이상·1만 자당 1개 이상·150개 이하; 제목 수준 2단계 이상·연속(빈 단계 없음)·최대 5단계; 빈 제목 문단 ≤ max(2, 5%); 제목 평균 길이 ≤80자·100자 초과 제목 ≤10%; 굵은 짧은 문단(제목 스타일 없이 굵기만 준 것) ≤ 제목 수×1.5; 표 ≤50개이며 표 글자 ≤ 전체 40%; 텍스트상자+프레임 ≤5; 목차 스타일 글자 ≤10%; Application에 PDF 변환기(Aspose/Sautin/Adobe 등) 없음.

1단계 탈락 사유(중복 집계):

| 사유 | ko | en |
|---|---|---|
| 글자수 | 143 | 63 |
| 제목<5 | 179 | 403 |
| 수준<2 | 179 | 440 |
| 수준불연속 | 9 | 100 |
| 빈제목 | 7 | 96 |
| 긴제목 | 7 | 74 |
| 굵은가짜제목 | 104 | 310 |
| 표과다 | 155 | 197 |
| 텍스트상자 | 8 | 53 |
| 목차과다 | 0 | 6 |
| 제목밀도 | 0 | 15 |
| PDF변환앱 | 0 | 0 |

**2단계 기준(Docling 변환 후)**: 변환 성공; md 빈 제목(`# `만 있는 줄) ≤1; md 제목 수가 python-docx 제목 수의 0.7~1.3배(스타일이 변환에서 제대로 살아남았는지); md 제목 수준 2종 이상; md 글자 수 범위; md 표 ≤30. **중복 제거**: 제목 목록 Jaccard ≥0.4면 하나만. **계열 제한**: NIST HB44 계열은 2개, `word_*` 계열 2개, 호주 법령(legal_judicial) 3개까지.

## 3. 한국어가 3개뿐인 이유

- 한국어 197개 중 179개(91%)는 Word 제목 문단이 5개 미만이다(실측). 1단계 통과는 4개, 그중 표 52개짜리 `ko_cb8b7087`이 2단계(md 표 ≤30)에서 빠져 최종 3개다. 굵은 짧은 문단만 있는 한국어 문서가 많아, 제목을 굵은 글씨로만 표현했을 가능성이 높다(앞선 실측과 같은 패턴, 개별 문서를 열어 확인하진 않음).
- 기준을 더 풀면 한국어가 늘어나지만 "스타일이 깔끔한 문서"라는 조건과 충돌한다. 한국어를 늘리려면 굵은 문단→제목 heuristic이 필요하다(결정 필요).
- 한국어 3개 중 `docx_7a16142d`(48,508자), `docx_5bba8969`(11,088자, vectara 하한 17k보다 짧음), `dotpad_kor`(12,269자, 표 7개)로, 한국어 문서는 vectara보다 짧은 편이다. 앞선 후보였던 `ko_cb8b7087`은 Docling md 표가 52개라 이번 기준(≤30)에서 탈락했다.

## 4. vectara PDF와 비교 (md 기준)

| 지표 | vectara PDF 50개 | 이번 docx 50개 |
|---|---|---|
| 글자 수 중앙값 | 58,359 | 47,054 |
| 글자 수 10~90분위 | 24,982~101,048 | 27,308~95,501 |
| 제목 줄 수 중앙값 (최소~최대) | 17.5 (1~42) | 31.5 (10~144) |
| 표 수 중앙값 (최소~최대) | 1.0 (0~15) | 3.0 (0~20) |
| 표 없는 문서 | 18 | 15 |

## 5. 최종 50개

| 파일 | 언어 | 유형 | 주제 | md 글자 수 | 제목 수 | 제목 수준 | 표 수 | 굵은 짧은 문단 | md 빈 제목 |
|---|---|---|---|---|---|---|---|---|---|
| 646_3ce5aca4.docx | en | administrative | government | 59,987 | 31 | 1,2,3 | 1 | 3 | 1 |
| 7cecaa9a-f4e2-461c-8335-7818f5e2a47d_0a791696.docx | en | administrative | government | 38,941 | 20 | 1,2 | 0 | 4 | 0 |
| GetFile_95daf995.docx | en | administrative | technology | 41,213 | 45 | 1,2,3 | 4 | 2 | 0 |
| download_8411204e.docx | en | administrative | government | 35,202 | 32 | 1,2,3,4 | 0 | 0 | 0 |
| 1922_81754fa4.docx | en | educational | education | 49,682 | 23 | 2,3 | 0 | 1 | 0 |
| 269_304da377.docx | en | educational | education | 77,187 | 33 | 1,2 | 5 | 0 | 0 |
| 30277416490135_0af6aee7.docx | en | educational | education | 104,218 | 115 | 1,2,3,4 | 6 | 0 | 0 |
| 3_c2251816.docx | en | educational | education | 32,377 | 22 | 1,2 | 1 | 8 | 0 |
| Research-Your-Subject-Chapter-8-Legalities-by-Joan | en | educational | education | 28,861 | 10 | 1,2 | 0 | 14 | 0 |
| RetrieveFile_4e19e993.docx | en | educational | technology | 65,642 | 30 | 1,2,3 | 5 | 22 | 0 |
| RetrieveFile_9fc3b75c.docx | en | educational | technology | 95,661 | 58 | 1,2,3,4 | 14 | 13 | 0 |
| SAP_NetWeaver_on_Windows_Azure_Virtual_Machine_Gui | en | educational | technology | 45,770 | 26 | 1,2,3 | 1 | 30 | 0 |
| c25cc72982e278864122d1a1e40409de7dcebada_14591076. | en | educational | education | 47,162 | 46 | 1,2,3,4 | 0 | 6 | 0 |
| content_40727ad8.docx | en | educational | finance | 92,983 | 15 | 1,2 | 8 | 1 | 0 |
| content_43003ad5.docx | en | educational | education | 30,154 | 11 | 1,2,3 | 0 | 1 | 0 |
| 2024_Bylaws_of_the_Greater_Folk_Polk_Area_REALTORS | en | legal | government | 71,168 | 19 | 1,2 | 0 | 0 | 0 |
| Births_Deaths_and_Marriages_Registration_Act_1998_ | en | legal | legal_judicial | 49,549 | 99 | 1,2,3,4,5 | 3 | 0 | 1 |
| Building_And_Construction_Industry_Training_Fund_A | en | legal | government | 46,946 | 49 | 1,2,3,4,5 | 6 | 0 | 0 |
| Court_Security_and_Custodial_Services_Regulations_ | en | legal | legal_judicial | 27,753 | 38 | 1,2,3,4,5 | 4 | 2 | 0 |
| District_Court_of_Western_Australia_Act_1969_Compa | en | legal | legal_judicial | 91,527 | 106 | 1,2,3,4,5 | 8 | 1 | 0 |
| Marketing_of_Potatoes_Act_1946_Compare__05-b0-02__ | en | legal | environment | 93,623 | 69 | 1,2,3,4,5 | 4 | 0 | 0 |
| content_6762153f.docx | en | legal | education | 94,067 | 26 | 1,2 | 20 | 21 | 0 |
| uwa-statute_f40fd8ab.docx | en | legal | education | 100,534 | 109 | 1,2 | 2 | 4 | 0 |
| 20230626103037_71_Code_of_Conduct_-_Staff_b5f5b16d | en | policies | education | 29,242 | 21 | 1,2,3,4 | 2 | 9 | 0 |
| 638644879362000000_94f8ff94.docx | en | policies | healthcare | 70,545 | 144 | 1,2 | 1 | 28 | 1 |
| A-6040_f6a48c82.docx | en | policies | finance | 35,807 | 45 | 1,2,3,4 | 15 | 9 | 0 |
| DisplayAttachment_8f4d354f.docx | en | policies | education | 29,337 | 85 | 1,2,3 | 0 | 3 | 0 |
| bsudatastream_2d8e60f2.docx | en | policies | education | 27,259 | 18 | 1,2,3 | 3 | 3 | 0 |
| download_86e0da07.docx | en | policies | finance | 64,522 | 45 | 1,2 | 7 | 18 | 0 |
| framewks-2015-hss-va-usgovt-ada_4136ed37.docx | en | policies | education | 96,319 | 140 | 2,3 | 2 | 183 | 0 |
| content_a8a937eb.docx | en | reference | healthcare | 36,894 | 14 | 1,2 | 5 | 0 | 0 |
| download_a9268ea6.docx | en | reference | government | 39,793 | 41 | 1,2,3,4 | 0 | 7 | 0 |
| download_c9b139c5.docx | en | reference | government | 41,537 | 39 | 1,2,3,4 | 0 | 1 | 0 |
| viewcontent_6f76e887.docx | en | reference | technology | 48,700 | 26 | 1,2,3 | 5 | 2 | 0 |
| word_3a74e2fe.docx | en | reference | government | 110,419 | 142 | 2,3,4,5 | 3 | 12 | 0 |
| word_863bea9b.docx | en | reference | healthcare | 87,350 | 108 | 2,3,4,5 | 4 | 12 | 0 |
| 2148_2ed8567e.docx | en | reports | nonprofit | 53,658 | 25 | 1,2 | 3 | 0 | 1 |
| AgResults-Vietnam-Sustainability-Assessment-Report | en | reports | environment | 71,524 | 25 | 1,2,3 | 9 | 1 | 1 |
| content_ea48c844.docx | en | reports | healthcare | 75,761 | 24 | 1,2 | 0 | 1 | 0 |
| docx_9a1e2897.docx | en | reports | government | 36,620 | 24 | 1,2,3 | 8 | 7 | 0 |
| download_18e7c9f5.docx | en | reports | government | 38,020 | 20 | 1,2,3,4 | 11 | 13 | 0 |
| download_3db37624.docx | en | reports | healthcare | 25,600 | 16 | 1,2,3 | 0 | 8 | 0 |
| impact-report-2023-plain-text_2a75791b.docx | en | reports | environment | 44,663 | 85 | 1,2,3,4 | 0 | 8 | 0 |
| 3-38-25-HB44-20241024_42b31d71.docx | en | technical | technology | 40,278 | 56 | 1,2,3,4 | 3 | 2 | 0 |
| 5-56a-15-hb44-final_f47d3e10.docx | en | technical | environment | 33,226 | 17 | 1,2,3,4 | 5 | 2 | 0 |
| RetrieveFile_10b2e807.docx | en | technical | technology | 78,958 | 66 | 1,2,3,4 | 14 | 0 | 1 |
| e2-dlqor-user-guide_055ecd14.docx | en | technical | environment | 24,998 | 14 | 1,2 | 2 | 5 | 1 |
| docx_5bba8969.docx | ko | educational | education | 11,377 | 22 | 1,2 | 0 | 11 | 1 |
| docx_7a16142d.docx | ko | educational | education | 49,938 | 51 | 1,2,3 | 0 | 46 | 0 |
| dotpad_kor_2711f7c9.docx | ko | technical | technology | 14,368 | 24 | 2,3,4 | 7 | 25 | 0 |

유형 분포: educational 13, technical 5, legal 8, reference 6, administrative 4, policies 7, reports 7.

## 6. 변환 결과 예시 (Docling md 머리)

### ko - docx_7a16142d.md

```markdown
# 소속감, 존재감, 그리고 자아 형성

호주 조기 학습 제도

# 목차

머리말 										3

아동 학습을 위한 비전								4

제도 요소								 	6

아동 학습  									6

조기 아동 교육									6

원칙  										7

업무										9

출생에서 5세까지의 어린이를 위한 학습 결과 					17

결과 1: 어린이는 강한 정체성을 가지고 있다 18

어린이는 안전하고 보호받고 지원되고 있음을 느낀다  				18

어린이는 자율성, 상호 의존, 회복력 및 주체성이 새롭게 발달한다  		19

어린이는 지식과 자신감에 기초한 자아 정체성이 발달한다  			20

어린이는 타인과의 관계에서 돌봄과 공감과 존중으로 상호작용하는 것을	배운다  20

결과 2: 어린이는 자신의 세계와 연결되고 그 세계에 기여한다 21

어린이는 그룹과 공동체에 대한 소속감, 그리고 적극적인 공동체 참여를 위해 		 필요한 상호 권리 및 책임에 대한 이해를 발달시킨다  				22

어린이는 다양성에 대해 존중하는 태도로 반응한다  				23

어린이는 공정성을 인식하기 시작한다  						23

어린이는 사회적으로 책임성을 가지게 되고 환경에 대해 존중하는 태도를 보인다 	24

결과 3: 어린이는 강한 복지감을 가진다 24

어린이는 자신의 사회적 정서적 복지에서 강해진다  				25

어린이는 자신의 건강과 신체적 복지에 대한 책임성을 증대시킨다  		26

결과 4: 어린이는 자신감이 있고 몰입하는 학습자이다 27

어린이는 호기심, 협동, 자신감, 창의성, 노력, 열의, 인내, 상상력 및 성찰 등 	28	 학습을 위한 기질을 발달시킨다

어린이는 문제 해결, 문의, 실험, 가정 설정, 연구 및 조사와 같은 다양한 기술과 	29	 절차를 발달시킨다

어린이는 한 상황에서 배운 것을 다른 상황으로 이동 적용시킨다  			29

어린이는 사람과 장소, 테크놀러지 및 자연적 가공적 재료와 연결시켜 자신의 학습 30	 자원을 제공한다

결과 5: 어린이는 효과적인 의사소통자이다 31

어린이는 다양한 목적을 위해 언어적 비언어적으로 타인과 상호작용을 한다  	32

어린이는 다양한 텍스트에 관여하고 이들 텍스트에서 의미를 파악한다  		33

어린이는 다양한 미디어를 사용하여 아이디어를 표현하고 의미를 창출한다  	34

어린이는 부호와 패턴 시스템이 작용하는 법을
```

### ko - dotpad_kor_2711f7c9.md

```markdown
닷패드 320 (DPA 320A)
사용자 매뉴얼

(08507)

서울특별시 금천구 가산디지털1로 146,

403호(대륭테크노타운 22차)

전화번호: 02-864-1113

팩스: 08-864-1989

이메일: inquiry@dotincorp.com

홈페이지: www.dotincorp.com

# 목차

목차	2

1. 닷패드 320 (DPA320A) 소개	3

1.1 인사말	3

1.2 닷패드 320 이란?	3

1.3. 닷패드 320 외형	4

1.4 하드웨어 사양	7

1.5 통신	8

2. 닷패드 320 기본 기능	9

2.1 닷패드 320 LED 신호	9

2.2 닷패드 320의 진동 신호를 통한 장치 정보	9

2.3 배터리 잔량 확인	10

2.4 충전 상태 확인	10

3. 닷패드 320 사용하기	10

3.1 닷 캔버스와 함께 닷패드 320 사용하기	11

3.2 보이스오버와 함께 닷패드 320 사용하기	11

3.3 NVDA와 함께 닷패드 320 사용하기	11

4. 취급 및 안전 주의사항	12

5. 고객 지원	16

6 제품 인증	16

## 1. 닷패드 320 (DPA320A) 소개

### 1.1 인사말

닷패드 320을 선택해 주셔서 감사합니다. 닷패드 320은 시각 정보에 대한 접근성을 높이고 시각장애 사용자들이 더 효율적으로 작업할 수 있도록, 생생한 촉각 그래픽과 멀티라인 점자 경험을 제공합니다. 주식회사 닷은 닷패드 320과 함께 보조 기술의 새로운 발전을 이끌어 나가기 위해 정진하겠습니다.

### 1.2 닷패드 320 이란?

닷패드 320은 점자 텍스트와 촉각 그래픽을 동시에 표시할 수 있는 디지털 촉각 정보 디스플레이입니다. 다양한 시각적 콘텐츠는 닷패드 320에 촉각 형식으로 표현되며 사용자는 복잡한 정보를 더 쉽게 이해하고 상호작용할 수 있습니다. 또한, 멀티라인 점자를 지원하여 한 화면에서 여러 줄의 점자를 표시할 수 있어 상세한 내용을 더욱 효과적으로 파악할 수 있습니다.

촉각 점자 디스플레이인 닷패드 320은 애플의 보이스오버, 윈도우 NVDA와 같은 스크린리더(화면 낭독기)와 호환됩니다. 또한 ㈜닷에서 직접 개발한 소프트웨어 서비스에 닷패드 320을 연결하여 사용 가능합니다. 이외에도 닷의 SDK(소프트웨어 개발 키트)를 활용하여 개발자 커뮤니티에서 개발한 애플리케이션을 닷패드 320과 함께 사용할 수 있습니다.

#### .1.2.1 닷패드 320 
```

### en - GetFile_95daf995.md

```markdown
# VQEG meeting minutes

Dates: Feb 23-27, 2015

Host: Intel.

Location: Santa Clara, CA, USA

Santa Clara Marriott Hotel , 2700 Mission College Blvd., Feb 23-25, 2015

Intel Corporation, 2200 Mission College Blvd., Feb 26-27, 2015

Participants: See Section Participants at the end the document

Monday, February 23, 2015

Presentations Expected:

OPTICOM (P.NATS / AVHD, 30 min, Chris)

Qualcomm (image quality, 30 min, James)

British Sky Broadcasting (AVHD, 20 min, Florence) challenges faced when doing subjective testing for OTT streaming

Acreo  and UPM (AVHD, 30 min, Samira & Kjell)

Acreo (statistical analysis,  15 min, Kjell)

Netflix (AVHD) may be able to present their needs/requirements which should be relevant for http streaming

AGH University (AVHD, 30 min, Lucjan)

AGH University (MOAVI, Mikolaj)

AGH University (QART, Mikolaj & Lucjan)

Intel (Tuesday at 10:30am, on HEVC, 15 min, Mark)

UWS (UltraHD results,  10 min, Naeem)

IRCCyN (UltraHD, 20 min, Marcus)

## Project Updates

ILG —nothing to report

AVHD —(1) video only objective quality metrics, may interact with UHD, (2) adaptive streaming, of high interest, will consider P.NATS effort this week, and (3) audiovisual q
```

### en - Building_And_Construction_Industry_Training_Fund_And_Levy_Collection_Act_1990_Co_dc635b17.md

```markdown
Western Australia

Building and Construction Industry Training Fund and Levy Collection Act 1990

Compare between:

[04 Mar 2011, 03-a0-02] and [11 Jul 2011, 03-b0-02]

| | <!-- image --> | |
|----|------------------|----|
| | | |

Western Australia

Building and Construction Industry Training Fund and Levy Collection Act 1990

An Act to establish a fund to be used to improve the quality of training and to increase the number of skilled persons in the building and construction industry, to establish a Building and Construction Industry Training Board to administer the fund and to collect the building and construction industry training levy, and for connected purposes.

## Part 1 — Preliminary

##### 1.	Short title

This Act may be cited as the Building and Construction Industry Training Fund and Levy Collection Act 1990 1 .

##### 2.	Commencement

This Act shall come into operation on the day on which the Building and Construction Industry Training Levy Act 1990 comes into operation 1 .

##### 3.	Terms used

(1)	In this Act, unless the contrary intention appears —

authorised person means a person appointed under section 28(1);

Board means the Building and Construction Industry Tr
```

## 7. 한계

- **한국어 3개**(위 3절). 나머지는 영어이므로 한국어 검색 성능은 이 코퍼스로 평가할 수 없다.
- 법령(legal) 8개와 NIST/호주 법령류는 조문 구조가 반복적이라 검색 난이도가 일반 보고서와 다를 수 있다. "굵은 짧은 문단이 제목 수의 1.5배까지 허용"이라서 굵은 라벨이 많은 문서가 남았다(`framewks-2015…` 183개, `docx_7a16142d` 46개, `SAP_NetWeaver…` 30개 등). 그 굵은 문단은 Docling md에서 제목이 아니라 일반 문단이 된다.
- 목차 스타일 글자 비율만 걸렀고, 목차를 굵은 일반 문단으로 쓴 문서는 남아 있을 수 있다.
- 일부 문서는 md에 빈 제목이 1개 남아 있다(기준 ≤1). loader가 빈 제목도 `#`로 출력하는 것은 이번 범위 밖이라 고치지 않았다.
- vectara보다 제목 줄 수(중앙값 31.5 vs 17.5)와 표 수(3 vs 1)가 많고 글자 수 중앙값은 더 적다(47k vs 58k). "비슷한 규모"이지 같은 분포는 아니다.
- 유형 균형은 완전하지 않다(educational 13 vs administrative 4).
- 후보 풀이 유형별 100개 무작위 표본이라 "docx-corpus 전체에서 최선"이 아니라 "이 표본에서 통과한 것"이다. 영어를 더 받으면 선택 폭이 넓어진다.