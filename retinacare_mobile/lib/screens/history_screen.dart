import 'dart:convert';
import 'package:flutter/material.dart';
import '../core/theme/app_colors.dart';
import '../core/theme/app_typography.dart';
import '../widgets/neu_card.dart';
import '../widgets/neu_chip.dart';
import '../services/storage_service.dart';
import '../models/prediction_result.dart';
import 'results_screen.dart';

/// RetinaCare AI - Obsidian & Champagne Gold Scan History Screen
class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  List<PredictionResult> _allScans = [];
  bool _isLoading = true;
  int _selectedFilter = 0; // 0: All, 1: No DR, 2: Referable

  @override
  void initState() {
    super.initState();
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    setState(() => _isLoading = true);
    try {
      final list = await StorageService.getScanHistory();
      if (mounted) {
        setState(() {
          _allScans = list;
          _isLoading = false;
        });
      }
    } catch (_) {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  Future<void> _deleteScan(String id) async {
    await StorageService.deleteScan(id);
    _loadHistory();
  }

  Future<void> _confirmClearAll() async {
    final bool? confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: AppColors.surfaceElevated,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: Text('Clear Scan History', style: AppTypography.sectionTitle),
        content: Text(
          'Are you sure you want to delete all locally stored diagnostic records?',
          style: AppTypography.bodySmall,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: Text('Cancel', style: AppTypography.caption.copyWith(color: AppColors.textSecondary)),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: Text('Delete All', style: AppTypography.caption.copyWith(color: AppColors.drStage4, fontWeight: FontWeight.w600)),
          ),
        ],
      ),
    );

    if (confirm == true) {
      await StorageService.clearAllHistory();
      _loadHistory();
    }
  }

  List<PredictionResult> get _filteredScans {
    if (_selectedFilter == 1) {
      return _allScans.where((s) => s.classId == 0).toList();
    } else if (_selectedFilter == 2) {
      return _allScans.where((s) => s.classId >= 2).toList();
    }
    return _allScans;
  }

  @override
  Widget build(BuildContext context) {
    final displayedScans = _filteredScans;

    return Scaffold(
      backgroundColor: Colors.transparent,
      body: SafeArea(
        child: RefreshIndicator(
          onRefresh: _loadHistory,
          color: AppColors.primaryGold,
          backgroundColor: AppColors.surfaceElevated,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header
              Padding(
                padding: const EdgeInsets.fromLTRB(18, 14, 18, 0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('Scan History', style: AppTypography.screenTitle),
                          const SizedBox(height: 2),
                          Text(
                            '${_allScans.length} diagnostic ${_allScans.length == 1 ? "record" : "records"} stored on-device',
                            style: AppTypography.caption,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ],
                      ),
                    ),
                    if (_allScans.isNotEmpty)
                      IconButton(
                        icon: const Icon(Icons.delete_sweep_outlined, color: AppColors.textMuted, size: 20),
                        tooltip: 'Clear All',
                        onPressed: _confirmClearAll,
                      ),
                  ],
                ),
              ),
              const SizedBox(height: 12),

              // Filter Chips (Neumorphic Chips with Champagne Gold active state)
              SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                padding: const EdgeInsets.symmetric(horizontal: 18),
                child: Row(
                  children: [
                    NeuChip(
                      label: 'All (${_allScans.length})',
                      isSelected: _selectedFilter == 0,
                      onTap: () => setState(() => _selectedFilter = 0),
                    ),
                    const SizedBox(width: 8),
                    NeuChip(
                      label: 'No DR (${_allScans.where((s) => s.classId == 0).length})',
                      isSelected: _selectedFilter == 1,
                      activeColor: AppColors.drStage0,
                      onTap: () => setState(() => _selectedFilter = 1),
                    ),
                    const SizedBox(width: 8),
                    NeuChip(
                      label: 'Referable (${_allScans.where((s) => s.classId >= 2).length})',
                      isSelected: _selectedFilter == 2,
                      activeColor: AppColors.drStage3,
                      onTap: () => setState(() => _selectedFilter = 2),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 12),

              // Scan List
              Expanded(
                child: _isLoading
                    ? const Center(child: CircularProgressIndicator(color: AppColors.primaryGold, strokeWidth: 2))
                    : displayedScans.isEmpty
                        ? _buildEmptyState()
                        : ListView.builder(
                            padding: const EdgeInsets.fromLTRB(18, 0, 18, 90),
                            itemCount: displayedScans.length,
                            itemBuilder: (context, index) {
                              final scan = displayedScans[index];
                              return _buildHistoryCard(scan);
                            },
                          ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            NeuCard(
              isInset: true,
              padding: const EdgeInsets.all(16),
              borderRadius: 20,
              child: const Icon(
                Icons.history_rounded,
                size: 32,
                color: AppColors.textMuted,
              ),
            ),
            const SizedBox(height: 14),
            Text(
              _selectedFilter == 0 ? 'No Scans Yet' : 'No Matching Scans',
              style: AppTypography.cardTitle,
            ),
            const SizedBox(height: 6),
            Text(
              _selectedFilter == 0
                  ? 'Completed retinal scans will appear here automatically.'
                  : 'No records match the selected filter category.',
              style: AppTypography.bodySmall,
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHistoryCard(PredictionResult scan) {
    final Color stageColor = AppColors.getColorForStage(scan.classId);

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      child: Dismissible(
        key: Key(scan.id),
        direction: DismissDirection.endToStart,
        background: Container(
          alignment: Alignment.centerRight,
          padding: const EdgeInsets.only(right: 18),
          decoration: BoxDecoration(
            color: AppColors.drStage4.withValues(alpha: 0.2),
            borderRadius: BorderRadius.circular(14),
          ),
          child: const Icon(Icons.delete_outline_rounded, color: AppColors.drStage4, size: 22),
        ),
        onDismissed: (_) => _deleteScan(scan.id),
        child: NeuCard(
          padding: const EdgeInsets.symmetric(horizontal: 14.0, vertical: 12.0),
          borderRadius: 14.0,
          onTap: () {
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => ResultsScreen(result: scan)),
            );
          },
          child: Row(
            children: [
              // 42x42 thumbnail
              Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: AppColors.surfacePressed,
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: stageColor.withValues(alpha: 0.35)),
                ),
                child: scan.localImageBase64 != null && scan.localImageBase64!.startsWith('data:image')
                    ? ClipRRect(
                        borderRadius: BorderRadius.circular(9),
                        child: Image.memory(
                          base64Decode(scan.localImageBase64!.split(',').last),
                          fit: BoxFit.cover,
                        ),
                      )
                    : Icon(Icons.visibility_rounded, color: stageColor, size: 20),
              ),
              const SizedBox(width: 12),
              // Stage name and timestamp
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      scan.shortName,
                      style: AppTypography.cardTitle.copyWith(fontSize: 14),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 2),
                    Text(
                      scan.timestamp,
                      style: AppTypography.caption.copyWith(fontSize: 10.5),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              // Confidence and Stage Pill
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    '${scan.confidencePct.toStringAsFixed(1)}%',
                    style: AppTypography.monoValue.copyWith(
                      color: stageColor,
                      fontSize: 13,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1.5),
                    decoration: BoxDecoration(
                      color: stageColor.withValues(alpha: 0.12),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(
                      'Stage ${scan.classId}',
                      style: AppTypography.caption.copyWith(
                        color: stageColor,
                        fontSize: 9.5,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
