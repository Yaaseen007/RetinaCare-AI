import 'dart:convert';
import 'package:flutter/material.dart';
import '../core/theme/app_colors.dart';
import '../core/theme/app_typography.dart';
import '../widgets/neu_card.dart';
import '../widgets/neu_button.dart';
import '../widgets/neu_stat_card.dart';
import '../services/storage_service.dart';
import '../services/api_service.dart';
import '../models/prediction_result.dart';
import 'results_screen.dart';

/// RetinaCare AI - Obsidian Black & Champagne Gold Dashboard
class HomeScreen extends StatefulWidget {
  final VoidCallback onNavigateToScan;
  final VoidCallback onNavigateToHistory;

  const HomeScreen({
    super.key,
    required this.onNavigateToScan,
    required this.onNavigateToHistory,
  });

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  List<PredictionResult> _recentScans = [];
  bool _isLoadingScans = true;
  bool _isBackendConnected = false;

  @override
  void initState() {
    super.initState();
    _loadDashboardData();
  }

  Future<void> _loadDashboardData() async {
    setState(() => _isLoadingScans = true);
    try {
      final scans = await StorageService.getScanHistory();
      if (mounted) {
        setState(() {
          _recentScans = scans;
          _isLoadingScans = false;
        });
      }
    } catch (_) {
      if (mounted) setState(() => _isLoadingScans = false);
    }

    _verifyBackendConnection();
  }

  Future<void> _verifyBackendConnection() async {
    try {
      await ApiService.checkHealth();
      if (mounted) {
        setState(() => _isBackendConnected = true);
      }
    } catch (_) {
      if (mounted) {
        setState(() => _isBackendConnected = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final int totalScans = _recentScans.length;
    final int normalScans = _recentScans.where((s) => s.classId == 0).length;
    final int referableScans = _recentScans.where((s) => s.classId >= 2).length;

    return RefreshIndicator(
      onRefresh: _loadDashboardData,
      color: AppColors.primaryGold,
      backgroundColor: AppColors.surfaceElevated,
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(18, 12, 18, 90),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // A. Header
            _buildHeader(),
            const SizedBox(height: 18),

            // B. Main Scan Section (Elegant raised neumorphic card)
            _buildMainScanCard(),
            const SizedBox(height: 14),

            // C. Model Information
            _buildModelIndicator(),
            const SizedBox(height: 18),

            // D. Statistics (Three compact metrics)
            _buildStatistics(totalScans, normalScans, referableScans),
            const SizedBox(height: 22),

            // E. Recent Scans (Latest 2-3 items)
            _buildRecentScans(),
          ],
        ),
      ),
    );
  }

  // --------------------------------------------------------------------------
  // A. Header: RetinaCare AI logo, application name, compact connectivity indicator
  // --------------------------------------------------------------------------
  Widget _buildHeader() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Expanded(
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              NeuCard(
                padding: const EdgeInsets.all(8),
                borderRadius: 12,
                child: const Icon(
                  Icons.visibility_rounded,
                  color: AppColors.primaryGold,
                  size: 18,
                ),
              ),
              const SizedBox(width: 10),
              Flexible(
                child: Text(
                  'RetinaCare AI',
                  style: AppTypography.screenTitle.copyWith(fontSize: 19),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(width: 8),
        // Compact backend connectivity indicator (Inset pill)
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
          decoration: BoxDecoration(
            color: AppColors.surfacePressed,
            borderRadius: BorderRadius.circular(12),
            gradient: AppColors.insetGradient,
            border: Border.all(
              color: _isBackendConnected
                  ? AppColors.drStage0.withValues(alpha: 0.4)
                  : AppColors.subtleBorder,
              width: 0.8,
            ),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 6,
                height: 6,
                decoration: BoxDecoration(
                  color: _isBackendConnected ? AppColors.drStage0 : AppColors.mutedGold,
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 6),
              Text(
                _isBackendConnected ? 'Online' : 'Offline',
                style: AppTypography.caption.copyWith(
                  color: _isBackendConnected ? AppColors.drStage0 : AppColors.textMuted,
                  fontWeight: FontWeight.w600,
                  fontSize: 11,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  // --------------------------------------------------------------------------
  // B. Main Scan Section: Elegant raised card with champagne gold accents
  // --------------------------------------------------------------------------
  Widget _buildMainScanCard() {
    return NeuCard(
      padding: const EdgeInsets.all(20.0),
      borderRadius: 18.0,
      isGoldAccent: true,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: AppColors.surfacePressed,
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: AppColors.primaryGold.withValues(alpha: 0.3)),
                ),
                child: const Icon(
                  Icons.center_focus_strong_rounded,
                  color: AppColors.primaryGold,
                  size: 20,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  'Diabetic Retinopathy Screening',
                  style: AppTypography.sectionTitle.copyWith(fontSize: 16),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            'Upload a retinal fundus image for automated 5-stage ICDR clinical evaluation.',
            style: AppTypography.bodySmall,
          ),
          const SizedBox(height: 18),
          NeuButton(
            label: 'Start New Scan',
            icon: Icons.add_photo_alternate_outlined,
            onPressed: widget.onNavigateToScan,
          ),
        ],
      ),
    );
  }

  // --------------------------------------------------------------------------
  // C. Model Information: Compact single-line indicator
  // --------------------------------------------------------------------------
  Widget _buildModelIndicator() {
    return Center(
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 5,
            height: 5,
            decoration: const BoxDecoration(
              color: AppColors.primaryGold,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 7),
          Flexible(
            child: Text(
              'MobileNetV2 · Mobile AI Engine',
              style: AppTypography.caption.copyWith(
                color: AppColors.textSecondary,
                fontSize: 11.5,
                fontWeight: FontWeight.w500,
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }

  // --------------------------------------------------------------------------
  // D. Statistics: Three compact metrics backed by real local history
  // --------------------------------------------------------------------------
  Widget _buildStatistics(int total, int normal, int referable) {
    return Row(
      children: [
        Expanded(
          child: NeuStatCard(
            label: 'Total Scans',
            value: total.toString(),
            accentColor: AppColors.primaryGold,
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: NeuStatCard(
            label: 'No DR',
            value: normal.toString(),
            accentColor: AppColors.drStage0,
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: NeuStatCard(
            label: 'Referable',
            value: referable.toString(),
            accentColor: AppColors.drStage3,
          ),
        ),
      ],
    );
  }

  // --------------------------------------------------------------------------
  // E. Recent Scans: Latest 2 or 3 items in a clean neumorphic list
  // --------------------------------------------------------------------------
  Widget _buildRecentScans() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text('Recent Scans', style: AppTypography.sectionTitle),
            if (_recentScans.isNotEmpty)
              GestureDetector(
                onTap: widget.onNavigateToHistory,
                child: Text(
                  'View all',
                  style: AppTypography.caption.copyWith(
                    color: AppColors.primaryGold,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
          ],
        ),
        const SizedBox(height: 12),
        if (_isLoadingScans)
          const Center(
            child: Padding(
              padding: EdgeInsets.all(20.0),
              child: CircularProgressIndicator(strokeWidth: 2, color: AppColors.primaryGold),
            ),
          )
        else if (_recentScans.isEmpty)
          NeuCard(
            isInset: true,
            padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 16),
            borderRadius: 14,
            child: Center(
              child: Text(
                'No scans recorded yet',
                style: AppTypography.bodySmall.copyWith(color: AppColors.textMuted),
              ),
            ),
          )
        else
          NeuCard(
            padding: EdgeInsets.zero,
            borderRadius: 14,
            child: ListView.separated(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              itemCount: _recentScans.take(3).length,
              separatorBuilder: (context, index) => const Divider(
                height: 1,
                color: AppColors.subtleBorder,
              ),
              itemBuilder: (context, index) {
                final scan = _recentScans[index];
                return _buildRecentScanRow(scan);
              },
            ),
          ),
      ],
    );
  }

  Widget _buildRecentScanRow(PredictionResult scan) {
    final Color stageColor = AppColors.getColorForStage(scan.classId);

    return InkWell(
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => ResultsScreen(result: scan),
          ),
        );
      },
      borderRadius: BorderRadius.circular(14),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 11),
        child: Row(
          children: [
            // 36x36 thumbnail
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Container(
                width: 36,
                height: 36,
                decoration: BoxDecoration(
                  color: AppColors.surfacePressed,
                  border: Border.all(color: stageColor.withValues(alpha: 0.3)),
                ),
                child: scan.localImageBase64 != null && scan.localImageBase64!.startsWith('data:image')
                    ? Image.memory(
                        base64Decode(scan.localImageBase64!.split(',').last),
                        fit: BoxFit.cover,
                      )
                    : Icon(Icons.remove_red_eye_rounded, color: stageColor, size: 18),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    scan.shortName,
                    style: AppTypography.cardTitle.copyWith(fontSize: 13.5),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 2),
                  Text(
                    scan.timestamp,
                    style: AppTypography.caption.copyWith(fontSize: 10.5),
                  ),
                ],
              ),
            ),
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text(
                  '${scan.confidencePct.toStringAsFixed(1)}%',
                  style: AppTypography.monoValue.copyWith(
                    color: stageColor,
                    fontSize: 12.5,
                  ),
                ),
                Text(
                  'Stage ${scan.classId}',
                  style: AppTypography.caption.copyWith(
                    color: stageColor,
                    fontSize: 10,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
