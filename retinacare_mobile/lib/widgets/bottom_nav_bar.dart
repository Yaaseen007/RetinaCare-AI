import 'package:flutter/material.dart';
import 'neu_navigation_bar.dart';

/// RetinaCare AI - Backward compatible wrapper for NeuNavigationBar
class LiquidBottomNavBar extends StatelessWidget {
  final int currentIndex;
  final ValueChanged<int> onTap;

  const LiquidBottomNavBar({
    super.key,
    required this.currentIndex,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return NeuNavigationBar(
      currentIndex: currentIndex,
      onTap: onTap,
    );
  }
}
