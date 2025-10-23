# 🏛️ المستشار القانوني السعودي الذكي | Saudi Legal AI Advisor

<div align="center">

**نموذج لغوي متخصص في القانون السعودي باستخدام QLoRA Fine-Tuning**

**Specialized Arabic Legal Language Model for Saudi Law using QLoRA**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

المطور | Developer: **Mostafa Ahmed El Sayed**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/mostafathemar/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/mostafathemar)

</div>

---

## 📋 جدول المحتويات | Table of Contents

- [🎯 نظرة عامة | Overview](#-نظرة-عامة--overview)
- [✨ المميزات الرئيسية | Key Features](#-المميزات-الرئيسية--key-features)
- [⚙️ التقنيات المستخدمة | Tech Stack](#️-التقنيات-المستخدمة--tech-stack)
- [🚀 البدء السريع | Quick Start](#-البدء-السريع--quick-start)
- [🎓 تفاصيل التدريب | Training Details](#-تفاصيل-التدريب--training-details)
- [⚙️ ضبط المعاملات | Hyperparameter Tuning](#️-ضبط-المعاملات--hyperparameter-tuning)
- [📊 التقييم | Evaluation](#-التقييم--evaluation)
- [💻 الاستخدام | Usage](#-الاستخدام--usage)
- [🗺️ خارطة الطريق | Roadmap](#️-خارطة-الطريق--roadmap)
- [🤝 المساهمة | Contributing](#-المساهمة--contributing)
- [📄 الترخيص | License](#-الترخيص--license)
- [📞 التواصل | Contact](#-التواصل--contact)
- [🙏 شكر وتقدير | Acknowledgments](#-شكر-وتقدير--acknowledgments)

---

## 🎯 نظرة عامة | Overview

<table>
<tr>
<td width="50%" dir="rtl">

### العربية

هذا المشروع يقدم **مستشار قانوني ذكي متخصص في القانون السعودي** باستخدام تقنيات الذكاء الاصطناعي الحديثة. تم بناء النموذج على أساس **Mistral-7B-Instruct-v0.2** وتخصيصه باستخدام تقنية **QLoRA** لتحقيق أداء عالٍ مع موارد محدودة.

#### المشكلة

- قلة الموارد الذكية المتخصصة في القانون السعودي باللغة العربية
- صعوبة الوصول للاستشارات القانونية الدقيقة
- تكلفة الاستشارات القانونية التقليدية

#### الحل

نموذج لغوي متقدم قادر على:

- فهم المصطلحات القانونية السعودية المعقدة
- تقديم إجابات دقيقة ومبنية على الأنظمة واللوائح
- التفاعل بلغة عربية طبيعية وواضحة

</td>
<td width="50%">

### English

This project presents a **Saudi Law specialized AI legal advisor** using cutting-edge AI techniques. Built on **Mistral-7B-Instruct-v0.2** and fine-tuned using **QLoRA** technology for high performance with limited resources.

#### The Problem

- Lack of specialized AI resources for Saudi law in Arabic
- Difficulty accessing accurate legal consultations
- High cost of traditional legal consultations

#### The Solution

An advanced language model capable of:

- Understanding complex Saudi legal terminology
- Providing accurate answers based on regulations and laws
- Interacting in natural and clear Arabic language

</td>
</tr>
</table>

---

## ✨ المميزات الرئيسية | Key Features

<table>
<tr>
<td width="50%" dir="rtl">

### العربية

- ✅ **تخصص عميق**: تدريب متخصص على القوانين والأنظمة السعودية
- 🚀 **كفاءة عالية**: استخدام QLoRA لتدريب سريع على GPU محدودة
- 💬 **واجهة تفاعلية**: تطبيق Gradio جاهز للاستخدام المباشر
- 🎯 **دقة عالية**: إجابات موثوقة مبنية على النصوص القانونية
- 🔧 **قابل للتوسع**: بنية مرنة لإضافة ميزات جديدة (RAG, API)
- 📱 **متعدد المنصات**: يعمل على Colab, Local GPU, Cloud

</td>
<td width="50%">

### English

- ✅ **Deep Specialization**: Trained on Saudi laws and regulations
- 🚀 **High Efficiency**: QLoRA for fast training on limited GPU
- 💬 **Interactive Interface**: Ready-to-use Gradio application
- 🎯 **High Accuracy**: Reliable answers based on legal texts
- 🔧 **Scalable**: Flexible architecture for new features (RAG, API)
- 📱 **Multi-Platform**: Works on Colab, Local GPU, Cloud

</td>
</tr>
</table>

---

## ⚙️ التقنيات المستخدمة | Tech Stack

| الفئة \| Category | الأداة \| Tool | الإصدار \| Version | الوصف \| Description |
|:---|:---|:---:|:---|
| **النموذج الأساسي**<br>Base Model | Mistral-7B-Instruct-v0.2 | v0.2 | نموذج مفتوح المصدر متقدم<br>Advanced open-source model |
| **التخصيص**<br>Fine-Tuning | QLoRA + PEFT | Latest | تدريب فعال مع 4-bit quantization<br>Efficient training with 4-bit quantization |
| **معالجة البيانات**<br>Data Processing | Pandas, Datasets | Latest | تنظيف وتجهيز البيانات<br>Data cleaning and preparation |
| **التقييم**<br>Evaluation | ROUGE, BLEU, Cosine Similarity | - | قياس جودة الإجابات<br>Measuring answer quality |
| **الواجهة**<br>Interface | Gradio | 4.0+ | واجهة مستخدم تفاعلية<br>Interactive user interface |
| **التخزين**<br>Storage | Google Drive, HuggingFace Hub | - | حفظ واسترجاع النماذج<br>Model storage and retrieval |

---

## 🚀 البدء السريع | Quick Start

### المتطلبات الأساسية | Prerequisites
```bash
# متطلبات النظام | System Requirements
- Python 3.10+
- CUDA 12.1+ (للتدريب على GPU | For GPU training)
- 16GB+ RAM
- Google Colab (مجاني مع T4 GPU | Free with T4 GPU) أو | or
- Local GPU (RTX 3060 12GB أو أعلى | or higher)
```

### 1️⃣ التثبيت | Installation
```python
# تثبيت المكتبات الأساسية | Install core libraries
!pip install -q unsloth[cu121] bitsandbytes accelerate
!pip install -q peft trl datasets pandas gradio transformers

# التحقق من التثبيت | Verify installation
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
print(f"✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### 2️⃣ إعداد البيانات | Data Preparation
```python
# هيكل ملف البيانات | Data file structure
# train_data.jsonl
{
  "instruction": "ما هي عقوبة التزوير في النظام السعودي؟",
  "input": "",
  "output": "وفقاً للنظام السعودي، عقوبة التزوير هي..."
}
```

### 3️⃣ التدريب | Training
```python
# افتح Notebook التدريب | Open Training Notebook
# 01_FineTuning_Saudi_Law_Expert.ipynb

# تحميل النموذج الأساسي | Load base model
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# تطبيق QLoRA | Apply QLoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)

# بدء التدريب | Start training
trainer.train()
```

### 4️⃣ الاستخدام | Usage
```python
# افتح Notebook الاستخدام | Open Inference Notebook
# 02_Saudi_Law_AI_Chat.ipynb

# تحميل النموذج المُدرب | Load trained model
model = FastLanguageModel.from_pretrained(
    model_name="path/to/lora_adapter",
    max_seq_length=2048,
)

# إطلاق واجهة Gradio | Launch Gradio interface
import gradio as gr

demo = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="سؤالك القانوني | Your Legal Question"),
    outputs=gr.Textbox(label="الإجابة | Answer"),
    title="🏛️ المستشار القانوني السعودي"
)

demo.launch()
```

---

## 🎓 تفاصيل التدريب | Training Details

### لماذا QLoRA؟ | Why QLoRA?

<table>
<tr>
<td width="50%" dir="rtl">

### العربية

**QLoRA** تقنية ثورية تجمع بين:

- **Quantization**: تقليل دقة الأوزان إلى 4-bit
- **LoRA**: تدريب طبقات صغيرة فقط (Adapters)

#### الفوائد

- 💾 **توفير الذاكرة**: تقليل استخدام VRAM بنسبة 75%
- ⚡ **سرعة أعلى**: تدريب أسرع 3-4x
- 💰 **تكلفة أقل**: يعمل على GPU مجانية (Colab T4)
- 🎯 **نفس الجودة**: أداء مماثل للتدريب الكامل

</td>
<td width="50%">

### English

**QLoRA** is a revolutionary technique combining:

- **Quantization**: Reducing weight precision to 4-bit
- **LoRA**: Training only small layers (Adapters)

#### Benefits

- 💾 **Memory Efficient**: 75% less VRAM usage
- ⚡ **Faster**: 3-4x faster training
- 💰 **Cost-Effective**: Works on free GPU (Colab T4)
- 🎯 **Same Quality**: Performance comparable to full training

</td>
</tr>
</table>

### معمارية النموذج | Model Architecture
```
┌─────────────────────────────────────────┐
│   Mistral-7B-Instruct-v0.2 (Base)      │
│        (Frozen - 4bit Quantized)        │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────▼─────────┐
        │   LoRA Adapters   │
        │   (Trainable)     │
        │   - r=16          │
        │   - alpha=32      │
        │   - dropout=0.05  │
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │ Legal Saudi Model │
        │   (Fine-tuned)    │
        └───────────────────┘
```

---

## ⚙️ ضبط المعاملات | Hyperparameter Tuning

| المعامل \| Parameter | القيمة الموصى بها<br>Recommended | النطاق \| Range | التأثير \| Impact |
|:---|:---:|:---:|:---|
| `lora_r` | 16 | 8-64 | رتبة مصفوفة LoRA - زيادتها تحسن التعلم<br>LoRA rank - higher improves learning |
| `lora_alpha` | 32 | 16-128 | قوة الدمج - عادة ضعف r<br>Scaling factor - usually 2×r |
| `max_seq_length` | 2048 | 512-4096 | طول السياق - للنصوص الطويلة<br>Context length - for long texts |
| `num_train_epochs` | 3 | 1-10 | عدد الحقب - احذر من Overfitting<br>Training epochs - watch for overfitting |
| `learning_rate` | 2e-4 | 1e-5 - 5e-4 | معدل التعلم - الأهم للجودة<br>Learning rate - critical for quality |
| `per_device_batch_size` | 4 | 1-16 | حجم الدفعة - حسب VRAM<br>Batch size - depends on VRAM |
| `gradient_accumulation` | 4 | 1-32 | لمحاكاة batch أكبر<br>To simulate larger batches |

### مثال التكوين الأمثل | Optimal Configuration Example
```python
training_arguments = TrainingArguments(
    output_dir="./saudi_law_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    warmup_steps=100,
)
```

---

## 📊 التقييم | Evaluation

### طرق التقييم | Evaluation Methods

<table>
<tr>
<td width="50%" dir="rtl">

#### 1. التقييم الآلي | Automatic Evaluation

- **ROUGE Score**: قياس التشابه مع الإجابات المرجعية
- **BLEU Score**: تقييم جودة الترجمة والصياغة
- **Cosine Similarity**: تشابه الدلالات بين الإجابات

#### 2. التقييم اليدوي | Manual Evaluation

- دقة المعلومات القانونية
- وضوح الصياغة
- الشمولية والتفصيل

</td>
<td width="50%">

#### 1. Automatic Evaluation

- **ROUGE Score**: Similarity with reference answers
- **BLEU Score**: Translation and phrasing quality
- **Cosine Similarity**: Semantic similarity

#### 2. Manual Evaluation

- Legal information accuracy
- Clarity of expression
- Comprehensiveness and detail

</td>
</tr>
</table>

### مثال النتائج | Sample Results
```
Test Set Performance:
├── ROUGE-L: 0.82
├── BLEU Score: 0.76
├── Accuracy: 89%
└── Response Time: <2s
```

---

## 💻 الاستخدام | Usage

### الاستخدام عبر Gradio | Using Gradio Interface
```python
import gradio as gr

def ask_legal_question(question):
    prompt = f"""### السؤال:
{question}

### الإجابة:"""
    
    response = model.generate(prompt, max_length=512)
    return response

interface = gr.Interface(
    fn=ask_legal_question,
    inputs=gr.Textbox(
        label="اطرح سؤالك القانوني",
        placeholder="مثال: ما هي شروط الزواج في النظام السعودي؟"
    ),
    outputs=gr.Textbox(label="الإجابة"),
    title="🏛️ المستشار القانوني السعودي",
    description="مساعد ذكي متخصص في القانون السعودي"
)

interface.launch(share=True)
```

### الاستخدام عبر Python API | Using Python API
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# تحميل النموذج | Load model
model = AutoModelForCausalLM.from_pretrained("path/to/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/model")

# الاستعلام | Query
question = "ما هي عقوبة السرقة في النظام السعودي؟"
inputs = tokenizer(question, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
answer = tokenizer.decode(outputs[0])

print(answer)
```

---

## 🗺️ خارطة الطريق | Roadmap

<table>
<tr>
<td width="50%" dir="rtl">

### المرحلة القادمة (Q1 2025)

- [ ] **دمج RAG**: إضافة قاعدة معرفة موسعة
- [ ] **API RESTful**: واجهة برمجية للتطبيقات
- [ ] **تحويل الصوت لنص**: استشارات صوتية
- [ ] **نموذج أكبر**: ترقية إلى Mistral-13B

### المرحلة المتقدمة (Q2-Q3 2025)

- [ ] **Multi-turn Dialog**: محادثات متعددة الأدوار
- [ ] **Document Analysis**: تحليل المستندات القانونية
- [ ] **Mobile App**: تطبيق هاتف محمول
- [ ] **دعم دول خليجية أخرى**: توسيع التغطية

</td>
<td width="50%">

### Next Phase (Q1 2025)

- [ ] **RAG Integration**: Extended knowledge base
- [ ] **RESTful API**: Application programming interface
- [ ] **Speech-to-Text**: Voice consultations
- [ ] **Larger Model**: Upgrade to Mistral-13B

### Advanced Phase (Q2-Q3 2025)

- [ ] **Multi-turn Dialog**: Conversation context
- [ ] **Document Analysis**: Legal document processing
- [ ] **Mobile App**: Mobile application
- [ ] **GCC Coverage**: Expand to other Gulf countries

</td>
</tr>
</table>

---

## 🤝 المساهمة | Contributing

<table>
<tr>
<td width="50%" dir="rtl">

### العربية

نرحب بمساهماتكم! يمكنك المساهمة عبر:

1. **Fork** المشروع
2. إنشاء **Branch** جديد (`git checkout -b feature/AmazingFeature`)
3. **Commit** التغييرات (`git commit -m 'Add some AmazingFeature'`)
4. **Push** للـ Branch (`git push origin feature/AmazingFeature`)
5. فتح **Pull Request**

#### مجالات المساهمة

- إضافة بيانات تدريب جديدة
- تحسين دقة النموذج
- ترجمة الواجهة
- كتابة الوثائق
- اختبار وإصلاح الأخطاء

</td>
<td width="50%">

### English

We welcome your contributions! You can contribute by:

1. **Fork** the project
2. Create a new **Branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the Branch (`git push origin feature/AmazingFeature`)
5. Open a **Pull Request**

#### Contribution Areas

- Add new training data
- Improve model accuracy
- Translate interface
- Write documentation
- Test and fix bugs

</td>
</tr>
</table>

---

## 📄 الترخيص | License
```
MIT License

Copyright (c) 2025 Mostafa Ahmed El Sayed

يُسمح باستخدام وتعديل وتوزيع هذا المشروع بحرية
This project is free to use, modify, and distribute
```

---

## 📞 التواصل | Contact

<div align="center">

**Mostafa Ahmed El Sayed**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mostafathemar/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mostafathemar)

📧 **Email**: [mostafathemar@gmail.com]

🌐 **Website**: [https://www.linkedin.com/in/mostafathemar/]

</div>

---

## 🙏 شكر وتقدير | Acknowledgments

<table>
<tr>
<td width="50%" dir="rtl">

### العربية

- **Mistral AI** - للنموذج الأساسي المفتوح المصدر
- **Hugging Face** - لمكتبات Transformers و PEFT
- **Unsloth** - لتحسينات QLoRA
- **Google Colab** - لتوفير موارد GPU المجانية
- **المجتمع العربي للذكاء الاصطناعي** - للدعم والإلهام

</td>
<td width="50%">

### English

- **Mistral AI** - For the open-source base model
- **Hugging Face** - For Transformers and PEFT libraries
- **Unsloth** - For QLoRA optimizations
- **Google Colab** - For providing free GPU resources
- **Arabic AI Community** - For support and inspiration

</td>
</tr>
</table>

---

<div align="center">

### ⭐ إذا أعجبك المشروع، لا تنسَ إضافة نجمة!
### ⭐ If you like this project, don't forget to star it!

**صُنع بـ ❤️ في السعودية | Made with ❤️ in Saudi Arabia**

</div>
