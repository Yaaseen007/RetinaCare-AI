import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:retinacare_mobile/core/theme/app_theme.dart';
import 'package:retinacare_mobile/models/prediction_result.dart';
import 'package:retinacare_mobile/screens/home_screen.dart';
import 'package:retinacare_mobile/screens/scanner_screen.dart';
import 'package:retinacare_mobile/screens/results_screen.dart';
import 'package:retinacare_mobile/screens/history_screen.dart';
import 'package:retinacare_mobile/screens/assistant_screen.dart';
import 'package:retinacare_mobile/screens/profile_screen.dart';
import 'package:retinacare_mobile/widgets/neu_navigation_bar.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUpAll(() {
    GoogleFonts.config.allowRuntimeFetching = false;
    Directory('screenshots').createSync(recursive: true);
  });

  final sampleScan1 = PredictionResult(
    id: 'scan_001',
    modelName: 'MobileNetV2',
    modelId: 'mobilenetv2',
    architectureType: 'Mobile Edge CNN',
    backbone: 'MobileNetV2',
    trainingType: 'Transfer Learning',
    datasetName: 'APTOS 2019',
    classId: 0,
    className: 'No Diabetic Retinopathy',
    shortName: 'No DR',
    confidencePct: 98.2,
    latencyMs: 118.0,
    isLowConfidence: false,
    confidenceStatus: 'High Confidence',
    confidenceThreshold: 50.0,
    probabilities: {
      'No DR': 0.982,
      'Mild NPDR': 0.012,
      'Moderate NPDR': 0.004,
      'Severe NPDR': 0.001,
      'Proliferative DR': 0.001,
    },
    rawProbabilities: [0.982, 0.012, 0.004, 0.001, 0.001],
    description: 'No microvascular abnormalities or diabetic lesions detected.',
    recommendation: 'Annual routine screening recommended.',
    severityColor: '#10B981',
    disclaimer: 'Clinical decision-support screening aid.',
    timestamp: '31 Oct 2026, 14:15',
  );

  final sampleScan2 = PredictionResult(
    id: 'scan_002',
    modelName: 'MobileNetV2',
    modelId: 'mobilenetv2',
    architectureType: 'Mobile Edge CNN',
    backbone: 'MobileNetV2',
    trainingType: 'Transfer Learning',
    datasetName: 'APTOS 2019',
    classId: 2,
    className: 'Moderate NPDR',
    shortName: 'Moderate NPDR',
    confidencePct: 91.4,
    latencyMs: 142.0,
    isLowConfidence: false,
    confidenceStatus: 'High Confidence',
    confidenceThreshold: 50.0,
    probabilities: {
      'No DR': 0.02,
      'Mild NPDR': 0.05,
      'Moderate NPDR': 0.914,
      'Severe NPDR': 0.01,
      'Proliferative DR': 0.006,
    },
    rawProbabilities: [0.02, 0.05, 0.914, 0.01, 0.006],
    description:
        'Moderate Non-Proliferative Diabetic Retinopathy characterized by multiple microaneurysms, dot and blot hemorrhages, and venous loops.',
    recommendation:
        'Dilated fundus examination and OCT recommended within 2 to 4 months. Optimize glycemic and blood pressure control.',
    severityColor: '#F59E0B',
    disclaimer: 'Clinical decision-support screening aid. Must be confirmed by licensed retinal specialist.',
    timestamp: '31 Oct 2026, 15:45',
  );

  setUp(() {
    SharedPreferences.setMockInitialValues({
      'retinacare_scan_history': [
        sampleScan1.toJsonString(),
        sampleScan2.toJsonString(),
      ],
      'retinacare_clinician_name': 'Dr. Retinal Specialist',
      'retinacare_clinic_facility': 'GM University Ophthalmology Center',
    });
  });

  Future<void> captureWidget({
    required WidgetTester tester,
    required Widget child,
    required String fileName,
    int activeTabIndex = 0,
  }) async {
    final GlobalKey repaintKey = GlobalKey();

    tester.view.physicalSize = const Size(393, 852);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(
      MaterialApp(
        debugShowCheckedModeBanner: false,
        theme: AppTheme.darkTheme,
        home: RepaintBoundary(
          key: repaintKey,
          child: Scaffold(
            backgroundColor: const Color(0xFF171717),
            body: SafeArea(
              bottom: false,
              child: child,
            ),
            bottomNavigationBar: NeuNavigationBar(
              currentIndex: activeTabIndex,
              onTap: (_) {},
            ),
          ),
        ),
      ),
    );

    await tester.pump();
    await tester.pump(const Duration(milliseconds: 400));

    await tester.runAsync(() async {
      final boundary = repaintKey.currentContext!.findRenderObject() as RenderRepaintBoundary;
      final image = await boundary.toImage(pixelRatio: 1.0);
      final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
      final pngBytes = byteData!.buffer.asUint8List();

      final file = File('screenshots/$fileName');
      file.writeAsBytesSync(pngBytes);
    });
  }

  testWidgets('Capture Home Screen Screenshot', (WidgetTester tester) async {
    await captureWidget(
      tester: tester,
      child: HomeScreen(onNavigateToScan: () {}, onNavigateToHistory: () {}),
      fileName: 'home_screen.png',
      activeTabIndex: 0,
    );
  });

  testWidgets('Capture Scanner Screen Screenshot', (WidgetTester tester) async {
    await captureWidget(
      tester: tester,
      child: ScannerScreen(onScanCompleted: () {}),
      fileName: 'scanner_screen.png',
      activeTabIndex: 1,
    );
  });

  testWidgets('Capture Results Screen Screenshot', (WidgetTester tester) async {
    final GlobalKey repaintKey = GlobalKey();

    tester.view.physicalSize = const Size(393, 852);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(
      MaterialApp(
        debugShowCheckedModeBanner: false,
        theme: AppTheme.darkTheme,
        home: RepaintBoundary(
          key: repaintKey,
          child: ResultsScreen(result: sampleScan2),
        ),
      ),
    );

    await tester.pump();
    await tester.pump(const Duration(milliseconds: 400));

    await tester.runAsync(() async {
      final boundary = repaintKey.currentContext!.findRenderObject() as RenderRepaintBoundary;
      final image = await boundary.toImage(pixelRatio: 1.0);
      final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
      final pngBytes = byteData!.buffer.asUint8List();

      File('screenshots/results_screen.png').writeAsBytesSync(pngBytes);
    });
  });

  testWidgets('Capture History Screen Screenshot', (WidgetTester tester) async {
    await captureWidget(
      tester: tester,
      child: const HistoryScreen(),
      fileName: 'history_screen.png',
      activeTabIndex: 2,
    );
  });

  testWidgets('Capture AI Assistant Screen Screenshot', (WidgetTester tester) async {
    await captureWidget(
      tester: tester,
      child: const AssistantScreen(),
      fileName: 'assistant_screen.png',
      activeTabIndex: 3,
    );
  });

  testWidgets('Capture Settings Screen Screenshot', (WidgetTester tester) async {
    await captureWidget(
      tester: tester,
      child: const ProfileScreen(),
      fileName: 'settings_screen.png',
      activeTabIndex: 4,
    );
  });
}
