import '../models/chat_message.dart';

/// RetinaCare AI - Educational Assistant Engine
/// Provides medically structured educational information on Diabetic Retinopathy.
/// Note: Never provides medical prescriptions or clinical diagnoses.
class AiAssistantService {
  AiAssistantService._();

  static const String educationalDisclaimer =
      '⚠️ EDUCATIONAL NOTICE: RetinaCare AI Assistant provides clinical education and awareness only. It does not replace medical consultation, clinical diagnosis, or treatment plans from a licensed ophthalmologist or optometrist.';

  static final List<String> initialSuggestions = [
    'What is Mild NPDR?',
    'Explain Diabetic Retinopathy Stages',
    'What are Microaneurysms?',
    'How often should diabetics have eye exams?',
    'What is MobileNetV2 on mobile?',
  ];

  static Future<ChatMessage> processQuery(String userQuery) async {
    final clean = userQuery.toLowerCase().trim();

    // Simulate natural thinking delay
    await Future.delayed(const Duration(milliseconds: 650));

    if (clean.contains('mild') || clean.contains('stage 1')) {
      return ChatMessage(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        text:
            '**Mild Non-Proliferative Diabetic Retinopathy (Mild NPDR - Stage 1)**\n\n'
            '• **Pathology:** Earliest clinically visible stage characterized strictly by microaneurysms—tiny balloon-like outpouchings in weakened retinal capillaries.\n'
            '• **Symptoms:** Often asymptomatic. Patients rarely notice visual changes at this stage.\n'
            '• **Clinical Recommendation:** Strict glycemic control (HbA1c < 7.0%), blood pressure management, and a dilated eye examination every 12 months.\n\n'
            '$educationalDisclaimer',
        isUser: false,
        timestamp: DateTime.now(),
        topic: 'Stage 1: Mild NPDR',
        suggestions: [
          'What are Microaneurysms?',
          'How does blood sugar cause NPDR?',
          'What is Moderate NPDR?',
        ],
      );
    } else if (clean.contains('moderate') || clean.contains('stage 2')) {
      return ChatMessage(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        text:
            '**Moderate Non-Proliferative Diabetic Retinopathy (Moderate NPDR - Stage 2)**\n\n'
            '• **Pathology:** Capillary occlusion expands. Retinal microaneurysms multiply, accompanied by dot-and-blot hemorrhages, venous beading, and hard exudates (lipid deposits).\n'
            '• **Clinical Concern:** Retinal ischemia increases the risk of diabetic macular edema (DME).\n'
            '• **Screening Interval:** Comprehensive dilated exam every 6 months by an eye care specialist.\n\n'
            '$educationalDisclaimer',
        isUser: false,
        timestamp: DateTime.now(),
        topic: 'Stage 2: Moderate NPDR',
        suggestions: [
          'What are Hard Exudates?',
          'What is Diabetic Macular Edema (DME)?',
          'What is Severe NPDR?',
        ],
      );
    } else if (clean.contains('severe') || clean.contains('stage 3')) {
      return ChatMessage(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        text:
            '**Severe Non-Proliferative Diabetic Retinopathy (Severe NPDR - Stage 3)**\n\n'
            '• **Diagnostic Standard (4-2-1 Rule):**\n'
            '  - Diffuse intraretinal hemorrhages in all 4 quadrants, OR\n'
            '  - Definite venous beading in 2+ quadrants, OR\n'
            '  - Intraretinal microvascular abnormalities (IRMA) in 1+ quadrant.\n'
            '• **Urgency:** High risk (>50% in 1 year) of progressing to vision-threatening Proliferative DR.\n'
            '• **Action:** Prompt ophthalmology referral within 2 to 4 weeks.\n\n'
            '$educationalDisclaimer',
        isUser: false,
        timestamp: DateTime.now(),
        topic: 'Stage 3: Severe NPDR',
        suggestions: [
          'What is the 4-2-1 Rule?',
          'What is Proliferative DR?',
          'What treatments exist?',
        ],
      );
    } else if (clean.contains('proliferat') || clean.contains('pdr') || clean.contains('stage 4')) {
      return ChatMessage(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        text:
            '**Proliferative Diabetic Retinopathy (PDR - Stage 4)**\n\n'
            '• **Pathology:** Severe retinal ischemia triggers VEGF (Vascular Endothelial Growth Factor) release, causing neovascularization—abnormal, fragile blood vessels growing on the disc (NVD) or elsewhere (NVE).\n'
            '• **Risks:** Vitreous hemorrhage, tractional retinal detachment, and neovascular glaucoma.\n'
            '• **Treatments:** Panretinal photocoagulation (PRP laser), Anti-VEGF intravitreal injections (e.g. Aflibercept, Ranibizumab), or vitrectomy.\n'
            '• **Action:** Immediate urgent ophthalmology intervention.\n\n'
            '$educationalDisclaimer',
        isUser: false,
        timestamp: DateTime.now(),
        topic: 'Stage 4: Proliferative DR',
        suggestions: [
          'How do Anti-VEGF injections work?',
          'What is Panretinal Photocoagulation?',
          'What are early signs of vitreous hemorrhage?',
        ],
      );
    } else if (clean.contains('microaneurysm')) {
      return ChatMessage(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        text:
            '**What are Retinal Microaneurysms?**\n\n'
            '• Microaneurysms are microscopic saccular outpouchings (10–100 μm in diameter) of capillary walls caused by pericyte loss due to chronic hyperglycemia.\n'
            '• On fundus photographs, they appear as isolated, round red dots with sharp margins.\n'
            '• They are the earliest detectable hallmark of Diabetic Retinopathy.\n\n'
            '$educationalDisclaimer',
        isUser: false,
        timestamp: DateTime.now(),
        topic: 'Biomarkers',
        suggestions: [
          'What causes pericyte loss?',
          'What is the difference between microaneurysms and dot hemorrhages?',
        ],
      );
    } else if (clean.contains('mobilenet') || clean.contains('model') || clean.contains('architecture')) {
      return ChatMessage(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        text:
            '**RetinaCare AI Mobile Architecture (MobileNetV2)**\n\n'
            '• **Dedicated Mobile Role:** MobileNetV2 is exclusively assigned to the Flutter mobile application for low-power, edge-efficient inference.\n'
            '• **Parameters:** ~2.62 Million parameters (compact 26.2 MB checkpoint).\n'
            '• **Latency:** Typically 140–220 ms on mobile CPU, avoiding battery drain.\n'
            '• **Separation:** The website uses VGG16 exclusively, while the mobile client routes to `/api/predict/mobile`.\n\n'
            '$educationalDisclaimer',
        isUser: false,
        timestamp: DateTime.now(),
        topic: 'System Architecture',
        suggestions: [
          'Why is MobileNetV2 better for phones?',
          'How does the image preprocessing work?',
        ],
      );
    } else if (clean.contains('exam') || clean.contains('screen') || clean.contains('often')) {
      return ChatMessage(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        text:
            '**Recommended Screening Guidelines for Diabetic Patients**\n\n'
            '• **Type 1 Diabetes:** First dilated eye exam within 5 years of diagnosis, then annually.\n'
            '• **Type 2 Diabetes:** Immediate dilated eye exam at the time of diagnosis, then annually.\n'
            '• **During Pregnancy:** Women with pre-existing diabetes should be screened prior to pregnancy or in the first trimester.\n'
            '• **With Diagnosed DR:** 3 to 6-month intervals depending on DR severity stage.\n\n'
            '$educationalDisclaimer',
        isUser: false,
        timestamp: DateTime.now(),
        topic: 'Screening Guidelines',
        suggestions: [
          'Why screen immediately in Type 2 Diabetes?',
          'What is a dilated eye exam?',
        ],
      );
    } else {
      return ChatMessage(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        text:
            '**Diabetic Retinopathy Clinical Overview**\n\n'
            'Diabetic Retinopathy (DR) is a microvascular complication of diabetes caused by chronic high blood glucose levels damaging the light-sensitive retina.\n\n'
            'The International Clinical Diabetic Retinopathy (ICDR) scale grades severity into 5 stages:\n'
            '1. **Stage 0 - No DR:** Healthy fundus with no vascular abnormalities.\n'
            '2. **Stage 1 - Mild NPDR:** Microaneurysms only.\n'
            '3. **Stage 2 - Moderate NPDR:** Microaneurysms, dot-and-blot hemorrhages, venous beading.\n'
            '4. **Stage 3 - Severe NPDR:** 4-2-1 rule met (extensive hemorrhages, venous beading, IRMA).\n'
            '5. **Stage 4 - Proliferative DR:** Neovascularization, risk of vitreous hemorrhage.\n\n'
            '$educationalDisclaimer',
        isUser: false,
        timestamp: DateTime.now(),
        topic: 'General Overview',
        suggestions: [
          'What is Mild NPDR?',
          'What is Severe NPDR?',
          'How does MobileNetV2 analyze my scan?',
        ],
      );
    }
  }
}

