import 'package:flutter/material.dart';
import 'core/theme/app_theme.dart';
import 'screens/main_navigation_shell.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const RetinaCareMobileApp());
}

/// RetinaCare AI - Mobile Application
/// Dedicated MobileNetV2 Inference Client for Android
class RetinaCareMobileApp extends StatelessWidget {
  const RetinaCareMobileApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'RetinaCare AI',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.darkTheme,
      home: const MainNavigationShell(),
    );
  }
}
