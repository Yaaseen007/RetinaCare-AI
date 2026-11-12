import 'package:flutter/material.dart';
import '../core/theme/app_colors.dart';
import '../widgets/neu_navigation_bar.dart';
import 'home_screen.dart';
import 'scanner_screen.dart';
import 'history_screen.dart';
import 'assistant_screen.dart';
import 'profile_screen.dart';

/// RetinaCare AI - Main Persistent Navigation Shell with Neumorphic Bottom Bar
class MainNavigationShell extends StatefulWidget {
  final int initialIndex;

  const MainNavigationShell({
    super.key,
    this.initialIndex = 0,
  });

  @override
  State<MainNavigationShell> createState() => _MainNavigationShellState();
}

class _MainNavigationShellState extends State<MainNavigationShell> {
  late int _currentIndex;

  @override
  void initState() {
    super.initState();
    _currentIndex = widget.initialIndex;
  }

  void _onTabSelected(int index) {
    if (_currentIndex != index) {
      setState(() => _currentIndex = index);
    }
  }

  void navigateToTab(int index) {
    setState(() => _currentIndex = index);
  }

  @override
  Widget build(BuildContext context) {
    final List<Widget> screens = [
      HomeScreen(onNavigateToScan: () => _onTabSelected(1), onNavigateToHistory: () => _onTabSelected(2)),
      ScannerScreen(onScanCompleted: () => _onTabSelected(2)),
      const HistoryScreen(),
      const AssistantScreen(),
      const ProfileScreen(),
    ];

    return Scaffold(
      backgroundColor: AppColors.background,
      body: Container(
        color: AppColors.background,
        child: SafeArea(
          bottom: false,
          child: IndexedStack(
            index: _currentIndex,
            children: screens,
          ),
        ),
      ),
      bottomNavigationBar: NeuNavigationBar(
        currentIndex: _currentIndex,
        onTap: _onTabSelected,
      ),
    );
  }
}
