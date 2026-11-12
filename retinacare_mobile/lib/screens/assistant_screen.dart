import 'package:flutter/material.dart';
import '../core/theme/app_colors.dart';
import '../core/theme/app_typography.dart';
import '../widgets/neu_card.dart';
import '../widgets/neu_chip.dart';
import '../widgets/neu_icon_button.dart';
import '../widgets/neu_input_field.dart';
import '../models/chat_message.dart';
import '../services/ai_assistant_service.dart';

/// RetinaCare AI - Luxury Neumorphic Educational Assistant Screen
class AssistantScreen extends StatefulWidget {
  final String? initialQuery;

  const AssistantScreen({
    super.key,
    this.initialQuery,
  });

  @override
  State<AssistantScreen> createState() => _AssistantScreenState();
}

class _AssistantScreenState extends State<AssistantScreen> {
  final TextEditingController _inputCtrl = TextEditingController();
  final ScrollController _scrollCtrl = ScrollController();

  final List<ChatMessage> _messages = [];
  bool _isTyping = false;

  @override
  void initState() {
    super.initState();
    // Compact welcome message
    _messages.add(
      ChatMessage(
        id: 'welcome',
        text:
            '**Welcome to RetinaCare AI Educational Assistant.**\n\n'
            'Ask questions about Diabetic Retinopathy pathophysiology, ICDR stages, retinal lesions, and clinical screening guidelines.\n\n'
            '${AiAssistantService.educationalDisclaimer}',
        isUser: false,
        timestamp: DateTime.now(),
        suggestions: AiAssistantService.initialSuggestions,
      ),
    );

    if (widget.initialQuery != null && widget.initialQuery!.isNotEmpty) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        _handleSendMessage(widget.initialQuery!);
      });
    }
  }

  @override
  void dispose() {
    _inputCtrl.dispose();
    _scrollCtrl.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollCtrl.hasClients) {
        _scrollCtrl.animateTo(
          _scrollCtrl.position.maxScrollExtent,
          duration: const Duration(milliseconds: 250),
          curve: Curves.easeOutQuad,
        );
      }
    });
  }

  Future<void> _handleSendMessage(String query) async {
    final text = query.trim();
    if (text.isEmpty || _isTyping) return;

    _inputCtrl.clear();

    final userMsg = ChatMessage(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      text: text,
      isUser: true,
      timestamp: DateTime.now(),
    );

    setState(() {
      _messages.add(userMsg);
      _isTyping = true;
    });
    _scrollToBottom();

    final response = await AiAssistantService.processQuery(text);

    if (mounted) {
      setState(() {
        _messages.add(response);
        _isTyping = false;
      });
      _scrollToBottom();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      resizeToAvoidBottomInset: true,
      body: SafeArea(
        child: Column(
          children: [
            // Minimal Header with Champagne Gold Icon
            Padding(
              padding: const EdgeInsets.fromLTRB(18, 14, 18, 8),
              child: Row(
                children: [
                  NeuCard(
                    padding: const EdgeInsets.all(8),
                    borderRadius: 12,
                    child: const Icon(
                      Icons.auto_awesome_rounded,
                      color: AppColors.primaryGold,
                      size: 18,
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('AI Assistant', style: AppTypography.screenTitle.copyWith(fontSize: 18)),
                        Text(
                          'Clinical Education & Disease Guidance',
                          style: AppTypography.caption.copyWith(fontSize: 11),
                        ),
                      ],
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3.5),
                    decoration: BoxDecoration(
                      color: AppColors.surfacePressed,
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: AppColors.primaryGold.withValues(alpha: 0.3), width: 0.8),
                    ),
                    child: Text(
                      'Educational',
                      style: AppTypography.caption.copyWith(
                        color: AppColors.primaryGold,
                        fontSize: 10,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                ],
              ),
            ),

            // Small Inset Educational Disclaimer Banner
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 18.0, vertical: 4.0),
              child: NeuCard(
                isInset: true,
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                borderRadius: 10,
                child: Row(
                  children: [
                    const Icon(Icons.info_outline, size: 14, color: AppColors.mutedGold),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        'Educational only. Not a substitute for professional clinical diagnosis.',
                        style: AppTypography.caption.copyWith(fontSize: 10.5),
                      ),
                    ),
                  ],
                ),
              ),
            ),

            // Chat Messages List
            Expanded(
              child: ListView.builder(
                controller: _scrollCtrl,
                padding: const EdgeInsets.fromLTRB(18, 8, 18, 16),
                itemCount: _messages.length,
                itemBuilder: (context, index) {
                  final msg = _messages[index];
                  return _buildMessageItem(msg);
                },
              ),
            ),

            // Typing Indicator
            if (_isTyping)
              Padding(
                padding: const EdgeInsets.only(left: 20, bottom: 8),
                child: Row(
                  children: [
                    const SizedBox(
                      width: 12,
                      height: 12,
                      child: CircularProgressIndicator(strokeWidth: 1.5, color: AppColors.primaryGold),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      'Synthesizing clinical education...',
                      style: AppTypography.caption.copyWith(color: AppColors.primaryGold),
                    ),
                  ],
                ),
              ),

            // Bottom-fixed Inset Message Composer with Raised Send Button
            _buildInputBar(),
          ],
        ),
      ),
    );
  }

  Widget _buildMessageItem(ChatMessage msg) {
    if (msg.isUser) {
      // Inset user message surface
      return Align(
        alignment: Alignment.centerRight,
        child: Container(
          margin: const EdgeInsets.symmetric(vertical: 5),
          constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.8),
          child: NeuCard(
            isInset: true,
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            borderRadius: 14,
            child: Text(
              msg.text,
              style: AppTypography.body.copyWith(
                color: AppColors.textPrimary,
                fontSize: 13.5,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
        ),
      );
    }

    // Raised assistant message surface
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 6),
        constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.88),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            NeuCard(
              padding: const EdgeInsets.all(14),
              borderRadius: 14,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.shield_outlined, size: 13, color: AppColors.primaryGold),
                      const SizedBox(width: 4),
                      Text(
                        msg.topic ?? 'Educational Guidance',
                        style: AppTypography.caption.copyWith(
                          color: AppColors.primaryGold,
                          fontWeight: FontWeight.w600,
                          fontSize: 10.5,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  _buildFormattedMarkdownText(msg.text),
                ],
              ),
            ),
            if (msg.suggestions != null && msg.suggestions!.isNotEmpty) ...[
              const SizedBox(height: 8),
              Wrap(
                spacing: 6,
                runSpacing: 5,
                children: msg.suggestions!.map((s) {
                  return NeuChip(
                    label: s,
                    isSelected: false,
                    onTap: () => _handleSendMessage(s),
                  );
                }).toList(),
              ),
            ],
          ],
        ),
      ),
    );
  }

  /// Parses **bold** markers into formatted TextSpans, fixing literal ** issue.
  Widget _buildFormattedMarkdownText(String text) {
    final List<InlineSpan> spans = [];
    final RegExp boldExp = RegExp(r'\*\*(.*?)\*\*', dotAll: true);

    int lastMatchEnd = 0;
    for (final Match match in boldExp.allMatches(text)) {
      if (match.start > lastMatchEnd) {
        spans.add(TextSpan(
          text: text.substring(lastMatchEnd, match.start),
          style: AppTypography.body.copyWith(
            fontSize: 13,
            color: AppColors.textSecondary,
            height: 1.45,
          ),
        ));
      }

      final boldContent = match.group(1) ?? '';
      spans.add(TextSpan(
        text: boldContent,
        style: AppTypography.body.copyWith(
          fontSize: 13,
          fontWeight: FontWeight.w700,
          color: AppColors.textPrimary,
          height: 1.45,
        ),
      ));

      lastMatchEnd = match.end;
    }

    if (lastMatchEnd < text.length) {
      spans.add(TextSpan(
        text: text.substring(lastMatchEnd),
        style: AppTypography.body.copyWith(
          fontSize: 13,
          color: AppColors.textSecondary,
          height: 1.45,
        ),
      ));
    }

    return SelectableText.rich(
      TextSpan(children: spans),
    );
  }

  Widget _buildInputBar() {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 6, 16, 12),
      decoration: const BoxDecoration(
        color: AppColors.background,
        border: Border(top: BorderSide(color: AppColors.subtleBorder, width: 0.5)),
      ),
      child: Row(
        children: [
          Expanded(
            child: NeuInputField(
              controller: _inputCtrl,
              hintText: 'Ask an educational question...',
              onSubmitted: _handleSendMessage,
            ),
          ),
          const SizedBox(width: 10),
          NeuIconButton(
            icon: Icons.send_rounded,
            iconColor: AppColors.primaryGold,
            tooltip: 'Send Message',
            size: 46,
            onTap: () => _handleSendMessage(_inputCtrl.text),
          ),
        ],
      ),
    );
  }
}
