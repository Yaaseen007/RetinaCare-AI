import 'package:flutter/material.dart';

/// RetinaCare AI - Obsidian Black & Champagne Gold Neumorphism Design System
/// Luxury dark neumorphic palette engineered for clinical precision & elegance.
class AppColors {
  AppColors._();

  // --------------------------------------------------------------------------
  // BACKGROUND
  // --------------------------------------------------------------------------
  /// Obsidian Black: #171717
  static const Color background = Color(0xFF171717);
  static const Color obsidianBlack = Color(0xFF171717);

  // --------------------------------------------------------------------------
  // SURFACES
  // --------------------------------------------------------------------------
  /// Primary Surface: #202020
  static const Color surface = Color(0xFF202020);
  static const Color surfacePrimary = Color(0xFF202020);

  /// Secondary Surface: #242424
  static const Color surfaceSecondary = Color(0xFF242424);

  /// Elevated Surface: #292929
  static const Color surfaceElevated = Color(0xFF292929);

  /// Pressed / Inset Surface: #1C1C1C
  static const Color surfacePressed = Color(0xFF1C1C1C);

  // --------------------------------------------------------------------------
  // CHAMPAGNE GOLD (Primary Luxury Accents)
  // --------------------------------------------------------------------------
  /// Primary Gold: #C6A66B
  static const Color primaryGold = Color(0xFFC6A66B);

  /// Soft Gold: #D8BD8A
  static const Color softGold = Color(0xFFD8BD8A);

  /// Muted Gold: #9D8150
  static const Color mutedGold = Color(0xFF9D8150);

  /// Gold Highlight: #E6D3A9
  static const Color goldHighlight = Color(0xFFE6D3A9);

  // --------------------------------------------------------------------------
  // TYPOGRAPHY (Warm Ivory & Muted Tones)
  // --------------------------------------------------------------------------
  /// Primary Text: #F2E9D8 (Warm ivory)
  static const Color textPrimary = Color(0xFFF2E9D8);

  /// Secondary Text: #B7B0A4
  static const Color textSecondary = Color(0xFFB7B0A4);

  /// Muted Text: #817B72
  static const Color textMuted = Color(0xFF817B72);

  /// Disabled Text: #625E57
  static const Color textDisabled = Color(0xFF625E57);

  /// Inverted text for bright gold surfaces
  static const Color textInverse = Color(0xFF171717);

  // --------------------------------------------------------------------------
  // BORDERS
  // --------------------------------------------------------------------------
  /// Subtle Border: #38332B
  static const Color subtleBorder = Color(0xFF38332B);

  // --------------------------------------------------------------------------
  // NEUMORPHIC SHADOWS
  // --------------------------------------------------------------------------
  /// Dark Shadow: #0D0D0D
  static const Color darkShadow = Color(0xFF0D0D0D);

  /// Light Shadow: #303030
  static const Color lightShadow = Color(0xFF303030);

  // --------------------------------------------------------------------------
  // CLINICAL ICDR SEVERITY (MUST retain distinct clinical colors!)
  // --------------------------------------------------------------------------
  static const Color drStage0 = Color(0xFF10B981); // Emerald (No DR)
  static const Color drStage1 = Color(0xFF0EA5E9); // Sky Blue (Mild NPDR)
  static const Color drStage2 = Color(0xFFF59E0B); // Amber (Moderate NPDR)
  static const Color drStage3 = Color(0xFFF97316); // Orange (Severe NPDR)
  static const Color drStage4 = Color(0xFFEF4444); // Crimson (Proliferative DR)

  // --------------------------------------------------------------------------
  // GOLD GRADIENTS (Restrained Luxury Accents)
  // --------------------------------------------------------------------------
  static const LinearGradient goldButtonGradient = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [
      goldHighlight,
      primaryGold,
      mutedGold,
    ],
  );

  static const LinearGradient goldSubtleGradient = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [
      Color(0xFF332B1D),
      Color(0xFF221E17),
    ],
  );

  static const LinearGradient insetGradient = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [
      Color(0xFF141414),
      Color(0xFF1E1E1E),
    ],
  );

  static const LinearGradient raisedGradient = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [
      Color(0xFF252525),
      Color(0xFF1E1E1E),
    ],
  );

  // Backward compatibility aliases
  static const Color primaryCyan = primaryGold;
  static const Color secondaryBlue = softGold;
  static const Color violet = primaryGold;
  static const Color cyanAccent = primaryGold;
  static const Color violetAccent = softGold;
  static const Color blueAccent = softGold;
  static const Color backgroundSecondary = surfaceSecondary;
  static const Color backgroundElevated = surfaceElevated;
  static const Color glassBorder = subtleBorder;
  static const Color glassBorderSubtle = subtleBorder;
  static const Color glassSurface = surface;
  static const LinearGradient primaryGradient = goldButtonGradient;
  static const LinearGradient primaryButtonGradient = goldButtonGradient;
  static const LinearGradient liquidBackgroundGradient = LinearGradient(
    begin: Alignment.topCenter,
    end: Alignment.bottomCenter,
    colors: [
      Color(0xFF171717),
      Color(0xFF141414),
      Color(0xFF171717),
    ],
  );

  // --------------------------------------------------------------------------
  // SHADOW GENERATORS FOR NEUMORPHISM
  // --------------------------------------------------------------------------
  static List<BoxShadow> neuRaised({
    double offset = 4.0,
    double blur = 8.0,
  }) =>
      [
        BoxShadow(
          color: darkShadow.withValues(alpha: 0.9),
          offset: Offset(offset, offset),
          blurRadius: blur,
        ),
        BoxShadow(
          color: lightShadow.withValues(alpha: 0.55),
          offset: Offset(-offset, -offset),
          blurRadius: blur,
        ),
      ];

  static List<BoxShadow> neuRaisedSoft({
    double offset = 2.5,
    double blur = 5.0,
  }) =>
      [
        BoxShadow(
          color: darkShadow.withValues(alpha: 0.8),
          offset: Offset(offset, offset),
          blurRadius: blur,
        ),
        BoxShadow(
          color: lightShadow.withValues(alpha: 0.4),
          offset: Offset(-offset, -offset),
          blurRadius: blur,
        ),
      ];

  /// Clinical severity color mapping
  static Color getColorForStage(int classId) {
    switch (classId) {
      case 0:
        return drStage0;
      case 1:
        return drStage1;
      case 2:
        return drStage2;
      case 3:
        return drStage3;
      case 4:
        return drStage4;
      default:
        return primaryGold;
    }
  }
}
