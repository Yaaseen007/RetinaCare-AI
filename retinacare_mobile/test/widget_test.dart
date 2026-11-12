import 'package:flutter_test/flutter_test.dart';
import 'package:retinacare_mobile/main.dart';

void main() {
  testWidgets('RetinaCare AI app smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const RetinaCareMobileApp());
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 300));

    // Verify that the title and branding appear
    expect(find.text('RetinaCare AI'), findsWidgets);
    expect(find.text('Diabetic Retinopathy Screening'), findsOneWidget);
    expect(find.text('Start New Scan'), findsOneWidget);
    expect(find.text('MobileNetV2 · Mobile AI Engine'), findsWidgets);

    // Verify navigation tabs
    expect(find.text('Home'), findsOneWidget);
    expect(find.text('Scan'), findsOneWidget);
    expect(find.text('History'), findsOneWidget);
    expect(find.text('Assistant'), findsOneWidget);
    expect(find.text('Settings'), findsOneWidget);
  });
}
