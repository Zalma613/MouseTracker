package hu.neuro.tracker;

import ai.onnxruntime.*;
import com.formdev.flatlaf.FlatDarkLaf;
import org.bytedeco.ffmpeg.global.avutil;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.Timer;
import javax.swing.border.EmptyBorder;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.awt.event.*;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;
import java.awt.geom.RoundRectangle2D;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.List;

public class MouseLadderTracker extends JFrame {

    // --- Adatstruktúrák ---
    public static class KeypointDef {
        String name;
        Color color;
        boolean isPaw;

        public KeypointDef(String name, Color color, boolean isPaw) {
            this.name = name;
            this.color = color;
            this.isPaw = isPaw;
        }

        @Override
        public String toString() {
            return name + (isPaw ? "  [Mancs]" : "");
        }
    }

    public static class Point2D {
        double x, y, conf;
        public Point2D(double x, double y, double conf) {
            this.x = x;
            this.y = y;
            this.conf = conf;
        }
    }

    public static class FrameAnnotation {
        int frameIndex;
        Map<String, Point2D> keypoints = new LinkedHashMap<>();
        Rectangle boundingBox;
    }

    // --- Állapotváltozók ---
    private final List<KeypointDef> skeletonDefs = new ArrayList<>();
    private final List<int[]> skeletonBones = new ArrayList<>();
    private final Map<Integer, FrameAnnotation> annotations = new TreeMap<>();
    private final List<Double> ladderRungsX = new ArrayList<>();

    private double ladderLevelY = 350.0;
    private double faultThresholdY = 385.0;
    private double fallThresholdY = 430.0;
    private double ladderLengthCm = 100.0;

    // Videó változók
    private File currentVideoFile;
    private FFmpegFrameGrabber grabber;
    private final Java2DFrameConverter converter = new Java2DFrameConverter();
    private BufferedImage currentImage;
    private int currentFrameIdx = 0;
    private int totalFrames = 0;
    private double videoFps = 30.0;
    private Timer playTimer;
    private boolean isSliderProgrammaticUpdate = false;

    // UI komponensek
    private VideoCanvas canvas;
    private JSlider frameSlider;
    private JLabel statusLabel;
    private JLabel frameCounterLabel;
    private JProgressBar progressBar;
    private DefaultListModel<KeypointDef> kpListModel;
    private JList<KeypointDef> kpList;
    private JCheckBox chkAddRungMode;
    private JCheckBox chkShowLines;
    private JTextField txtLadderCm;
    private JButton btnPlayPause;

    // ONNX AI Modell
    private OrtEnvironment ortEnv;
    private OrtSession ortSession;

    public MouseLadderTracker() {
        super("Mouse Horizontal Ladder Tracker Studio - AI Pose & Kinematics");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1460, 900);
        setLocationRelativeTo(null);

        initDefaultSkeleton();
        initUI();
        initGlobalKeyBindings();
    }

    private void initDefaultSkeleton() {
        skeletonDefs.add(new KeypointDef("Fej_Orr", new Color(255, 75, 75), false));
        skeletonDefs.add(new KeypointDef("Gerinc_Nyak", new Color(255, 165, 0), false));
        skeletonDefs.add(new KeypointDef("Gerinc_Kozep", new Color(255, 220, 40), false));
        skeletonDefs.add(new KeypointDef("Farok_To", new Color(170, 255, 70), false));
        skeletonDefs.add(new KeypointDef("Farok_Veg", new Color(90, 210, 110), false));
        skeletonDefs.add(new KeypointDef("Mancs_Bal_Elso", new Color(0, 225, 255), true));
        skeletonDefs.add(new KeypointDef("Mancs_Jobb_Elso", new Color(40, 130, 255), true));
        skeletonDefs.add(new KeypointDef("Mancs_Bal_Hatso", new Color(230, 90, 255), true));
        skeletonDefs.add(new KeypointDef("Mancs_Jobb_Hatso", new Color(155, 65, 245), true));

        skeletonBones.add(new int[]{0, 1});
        skeletonBones.add(new int[]{1, 2});
        skeletonBones.add(new int[]{2, 3});
        skeletonBones.add(new int[]{3, 4});
        skeletonBones.add(new int[]{1, 5});
        skeletonBones.add(new int[]{1, 6});
        skeletonBones.add(new int[]{3, 7});
        skeletonBones.add(new int[]{3, 8});
    }

    private void initUI() {
        JPanel root = new JPanel(new BorderLayout(10, 10));
        root.setBorder(new EmptyBorder(10, 12, 10, 12));
        setContentPane(root);

        // --- Felső modern eszköztár ---
        JPanel topBar = new JPanel(new BorderLayout(10, 0));
        JPanel actionBtns = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 0));

        JButton btnOpenVideo = createStyledButton("1. Videó megnyitása", new Color(55, 90, 127));
        btnOpenVideo.addActionListener(e -> openVideo());

        JButton btnExportDataset = createStyledButton("2. Tanító adathalmaz mentése (YOLO)", new Color(70, 80, 95));
        btnExportDataset.addActionListener(e -> exportYoloDataset());

        JButton btnLoadModel = createStyledButton("3. AI Modell (.onnx) betöltése", new Color(110, 75, 145));
        btnLoadModel.addActionListener(e -> loadOnnxModel());

        JButton btnRunAnalysis = createStyledButton("4. Videó elemzése és CSV export", new Color(40, 145, 85));
        btnRunAnalysis.addActionListener(e -> runFullVideoAnalysis());

        actionBtns.add(btnOpenVideo);
        actionBtns.add(btnExportDataset);
        actionBtns.add(btnLoadModel);
        actionBtns.add(btnRunAnalysis);

        JLabel shortcutHint = new JLabel("Nyilak (←/→): ±1 frame  |  Fel/Le (↑/↓): ±10 frame  |  Bal klikk+húzás: Mozgatás  |  Jobb klikk: Törlés");
        shortcutHint.setForeground(new Color(170, 180, 195));
        shortcutHint.setFont(shortcutHint.getFont().deriveFont(Font.PLAIN, 12f));

        topBar.add(actionBtns, BorderLayout.WEST);
        topBar.add(shortcutHint, BorderLayout.EAST);
        root.add(topBar, BorderLayout.NORTH);

        // --- Középső interaktív videó vászon ---
        canvas = new VideoCanvas();
        root.add(canvas, BorderLayout.CENTER);

        // --- Jobb oldali vezérlőpanel ---
        JPanel sidePanel = new JPanel();
        sidePanel.setPreferredSize(new Dimension(340, 0));
        sidePanel.setLayout(new BorderLayout(8, 10));

        // 1. Testrészek / Csontváz lista (színes jelölőkkel)
        JPanel skelPanel = new JPanel(new BorderLayout(6, 6));
        skelPanel.setBorder(new TitledBorder("Testrészek (1-9 billentyű vagy kattintás)"));

        kpListModel = new DefaultListModel<>();
        skeletonDefs.forEach(kpListModel::addElement);
        kpList = new JList<>(kpListModel);
        kpList.setSelectedIndex(0);
        kpList.setFixedCellHeight(30);
        kpList.setCellRenderer(new DefaultListCellRenderer() {
            @Override
            public Component getListCellRendererComponent(JList<?> list, Object value, int index, boolean isSelected, boolean cellHasFocus) {
                JLabel lbl = (JLabel) super.getListCellRendererComponent(list, value, index, isSelected, cellHasFocus);
                KeypointDef def = (KeypointDef) value;
                FrameAnnotation fa = annotations.get(currentFrameIdx);
                boolean placed = (fa != null && fa.keypoints.containsKey(def.name));
                lbl.setText(String.format(" %d. %s %s", index + 1, def.toString(), placed ? "✓" : ""));
                lbl.setIcon(new ColorDotIcon(def.color, placed));
                return lbl;
            }
        });

        skelPanel.add(new JScrollPane(kpList), BorderLayout.CENTER);

        JPanel kpBtnPanel = new JPanel(new GridLayout(1, 2, 6, 6));
        JButton btnAddKp = new JButton("+ Új testrész");
        btnAddKp.addActionListener(e -> addCustomKeypoint());
        JButton btnClearFrame = new JButton("Képkocka törlése");
        btnClearFrame.addActionListener(e -> {
            annotations.remove(currentFrameIdx);
            kpList.repaint();
            canvas.repaint();
        });
        kpBtnPanel.add(btnAddKp);
        kpBtnPanel.add(btnClearFrame);
        skelPanel.add(kpBtnPanel, BorderLayout.SOUTH);

        sidePanel.add(skelPanel, BorderLayout.CENTER);

        // 2. Létra és Kalibráció panel
        JPanel ladderPanel = new JPanel();
        ladderPanel.setLayout(new BoxLayout(ladderPanel, BoxLayout.Y_AXIS));
        ladderPanel.setBorder(new TitledBorder("Létra és Vonalak (Húzd a vonalakat az egérrel!)"));

        chkShowLines = new JCheckBox("Határvonalak megjelenítése a videón", true);
        chkShowLines.addActionListener(e -> canvas.repaint());

        chkAddRungMode = new JCheckBox("Létrafok (Rung) lerakási mód", false);
        chkAddRungMode.setToolTipText("Ha be van pipálva, a bal klikk létrafokot tesz le, a jobb klikk törli.");

        JPanel calPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 6));
        calPanel.add(new JLabel("Létra valós hossza (cm):"));
        txtLadderCm = new JTextField(String.valueOf(ladderLengthCm), 6);
        calPanel.add(txtLadderCm);

        JButton btnClearRungs = new JButton("Összes létrafok törlése");
        btnClearRungs.setAlignmentX(Component.LEFT_ALIGNMENT);
        btnClearRungs.addActionListener(e -> {
            ladderRungsX.clear();
            canvas.repaint();
        });

        chkShowLines.setAlignmentX(Component.LEFT_ALIGNMENT);
        chkAddRungMode.setAlignmentX(Component.LEFT_ALIGNMENT);
        calPanel.setAlignmentX(Component.LEFT_ALIGNMENT);

        ladderPanel.add(chkShowLines);
        ladderPanel.add(Box.createVerticalStrut(4));
        ladderPanel.add(chkAddRungMode);
        ladderPanel.add(Box.createVerticalStrut(6));
        ladderPanel.add(calPanel);
        ladderPanel.add(Box.createVerticalStrut(4));
        ladderPanel.add(btnClearRungs);

        sidePanel.add(ladderPanel, BorderLayout.SOUTH);
        root.add(sidePanel, BorderLayout.EAST);

        // --- Alsó vezérlő és idősáv ---
        JPanel bottomPanel = new JPanel(new BorderLayout(10, 6));

        JPanel transportPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 0));
        JButton btnPrev10 = new JButton("<< -10");
        JButton btnPrev = new JButton("< -1");
        btnPlayPause = new JButton("▶ Lejátszás");
        JButton btnNext = new JButton("+1 >");
        JButton btnNext10 = new JButton("+10 >>");

        btnPrev10.addActionListener(e -> stepFrame(-10));
        btnPrev.addActionListener(e -> stepFrame(-1));
        btnPlayPause.addActionListener(e -> togglePlayPause());
        btnNext.addActionListener(e -> stepFrame(1));
        btnNext10.addActionListener(e -> stepFrame(10));

        frameCounterLabel = new JLabel(" 0 / 0  (0.00 s) ");
        frameCounterLabel.setFont(new Font(Font.MONOSPACED, Font.BOLD, 13));

        transportPanel.add(btnPrev10);
        transportPanel.add(btnPrev);
        transportPanel.add(btnPlayPause);
        transportPanel.add(btnNext);
        transportPanel.add(btnNext10);
        transportPanel.add(frameCounterLabel);

        frameSlider = new JSlider(0, 100, 0);
        frameSlider.setFocusable(false);
        frameSlider.addChangeListener(e -> {
            if (!isSliderProgrammaticUpdate && !frameSlider.getValueIsAdjusting()) {
                seekToFrame(frameSlider.getValue());
            } else if (!isSliderProgrammaticUpdate) {
                // Csúszka húzása közben is azonnal frissít
                seekToFrame(frameSlider.getValue());
            }
        });

        JPanel statusRow = new JPanel(new BorderLayout(10, 0));
        statusLabel = new JLabel(" Kész. Nyiss meg egy videót az 1-es gombbal!");
        progressBar = new JProgressBar(0, 100);
        progressBar.setPreferredSize(new Dimension(260, 18));
        progressBar.setStringPainted(true);
        progressBar.setVisible(false);

        statusRow.add(statusLabel, BorderLayout.CENTER);
        statusRow.add(progressBar, BorderLayout.EAST);

        bottomPanel.add(transportPanel, BorderLayout.WEST);
        bottomPanel.add(frameSlider, BorderLayout.CENTER);
        bottomPanel.add(statusRow, BorderLayout.SOUTH);

        root.add(bottomPanel, BorderLayout.SOUTH);

        playTimer = new Timer(33, e -> {
            if (currentFrameIdx < totalFrames - 1) {
                stepFrame(1);
            } else {
                togglePlayPause();
            }
        });
    }

    private JButton createStyledButton(String text, Color bg) {
        JButton b = new JButton(text);
        b.setBackground(bg);
        b.setForeground(Color.WHITE);
        b.setFocusPainted(false);
        b.setFont(b.getFont().deriveFont(Font.BOLD, 12.5f));
        return b;
    }

    // --- Globális billentyűparancsok (Nyilak, Space, Számgombok) ---
    private void initGlobalKeyBindings() {
        JRootPane rootPane = getRootPane();
        InputMap im = rootPane.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW);
        ActionMap am = rootPane.getActionMap();

        im.put(KeyStroke.getKeyStroke(KeyEvent.VK_LEFT, 0), "prevFrame");
        im.put(KeyStroke.getKeyStroke(KeyEvent.VK_RIGHT, 0), "nextFrame");
        im.put(KeyStroke.getKeyStroke(KeyEvent.VK_LEFT, InputEvent.SHIFT_DOWN_MASK), "prev10Frame");
        im.put(KeyStroke.getKeyStroke(KeyEvent.VK_RIGHT, InputEvent.SHIFT_DOWN_MASK), "next10Frame");
        im.put(KeyStroke.getKeyStroke(KeyEvent.VK_DOWN, 0), "prev10Frame");
        im.put(KeyStroke.getKeyStroke(KeyEvent.VK_UP, 0), "next10Frame");
        im.put(KeyStroke.getKeyStroke(KeyEvent.VK_SPACE, 0), "playPause");

        am.put("prevFrame", new AbstractAction() {
            @Override public void actionPerformed(ActionEvent e) {
                if (!(KeyboardFocusManager.getCurrentKeyboardFocusManager().getFocusOwner() instanceof JTextField)) {
                    stepFrame(-1);
                }
            }
        });
        am.put("nextFrame", new AbstractAction() {
            @Override public void actionPerformed(ActionEvent e) {
                if (!(KeyboardFocusManager.getCurrentKeyboardFocusManager().getFocusOwner() instanceof JTextField)) {
                    stepFrame(1);
                }
            }
        });
        am.put("prev10Frame", new AbstractAction() {
            @Override public void actionPerformed(ActionEvent e) {
                if (!(KeyboardFocusManager.getCurrentKeyboardFocusManager().getFocusOwner() instanceof JTextField)) {
                    stepFrame(-10);
                }
            }
        });
        am.put("next10Frame", new AbstractAction() {
            @Override public void actionPerformed(ActionEvent e) {
                if (!(KeyboardFocusManager.getCurrentKeyboardFocusManager().getFocusOwner() instanceof JTextField)) {
                    stepFrame(10);
                }
            }
        });
        am.put("playPause", new AbstractAction() {
            @Override public void actionPerformed(ActionEvent e) {
                if (!(KeyboardFocusManager.getCurrentKeyboardFocusManager().getFocusOwner() instanceof JTextField)) {
                    togglePlayPause();
                }
            }
        });

        // 1-9 gombok a testrészek gyors kiválasztásához
        for (int i = 0; i < 9; i++) {
            final int idx = i;
            im.put(KeyStroke.getKeyStroke(KeyEvent.VK_1 + i, 0), "selectKp_" + i);
            am.put("selectKp_" + i, new AbstractAction() {
                @Override public void actionPerformed(ActionEvent e) {
                    if (!(KeyboardFocusManager.getCurrentKeyboardFocusManager().getFocusOwner() instanceof JTextField)) {
                        if (idx < kpListModel.size()) kpList.setSelectedIndex(idx);
                    }
                }
            });
        }
    }

    private void togglePlayPause() {
        if (grabber == null) return;
        if (playTimer.isRunning()) {
            playTimer.stop();
            btnPlayPause.setText("▶ Lejátszás");
        } else {
            int delay = (int) Math.max(15, Math.round(1000.0 / videoFps));
            playTimer.setDelay(delay);
            playTimer.start();
            btnPlayPause.setText("⏸ Szünet");
        }
    }

    private void stepFrame(int delta) {
        if (grabber == null) return;
        int target = Math.max(0, Math.min(totalFrames - 1, currentFrameIdx + delta));
        if (target != currentFrameIdx) {
            seekToFrame(target);
        }
    }

    // --- Videó megnyitása és képkocka-olvasás ---
    private void openVideo() {
        if (playTimer.isRunning()) togglePlayPause();
        JFileChooser chooser = new JFileChooser();
        if (currentVideoFile != null) chooser.setCurrentDirectory(currentVideoFile.getParentFile());
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            currentVideoFile = chooser.getSelectedFile();
            try {
                if (grabber != null) grabber.close();
                grabber = new FFmpegFrameGrabber(currentVideoFile);
                grabber.start();
                totalFrames = grabber.getLengthInFrames();
                videoFps = grabber.getFrameRate() > 0 ? grabber.getFrameRate() : 30.0;

                // Alapértelmezett határvonalak a kép magasságához igazítva
                int h = grabber.getImageHeight();
                if (h > 0 && annotations.isEmpty()) {
                    ladderLevelY = h * 0.60;
                    faultThresholdY = h * 0.66;
                    fallThresholdY = h * 0.74;
                }

                isSliderProgrammaticUpdate = true;
                frameSlider.setMaximum(Math.max(1, totalFrames - 1));
                isSliderProgrammaticUpdate = false;

                seekToFrame(0);
                statusLabel.setText(String.format(" Videó betöltve: %s (%dx%d @ %.1f FPS)",
                        currentVideoFile.getName(), grabber.getImageWidth(), grabber.getImageHeight(), videoFps));
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this, "Hiba a videó megnyitásakor: " + ex.getMessage());
            }
        }
    }

    private synchronized void seekToFrame(int frameIdx) {
        if (grabber == null) return;
        try {
            currentFrameIdx = frameIdx;
            grabber.setVideoFrameNumber(frameIdx);
            Frame frame = grabber.grabImage();
            if (frame != null) {
                currentImage = converter.getBufferedImage(frame);
                isSliderProgrammaticUpdate = true;
                frameSlider.setValue(frameIdx);
                isSliderProgrammaticUpdate = false;

                frameCounterLabel.setText(String.format(" %d / %d  (%.2f s) ",
                        frameIdx, Math.max(0, totalFrames - 1), frameIdx / videoFps));
                kpList.repaint();
                canvas.repaint();
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    private void addCustomKeypoint() {
        String name = JOptionPane.showInputDialog(this, "Új testrész neve (pl. Gerinc_Csipo):");
        if (name != null && !name.trim().isEmpty()) {
            int isPawOpt = JOptionPane.showConfirmDialog(this,
                    "Ez a pont egy végtag/mancs (lépésszámláláshoz és mellélépéshez)?",
                    "Típus", JOptionPane.YES_NO_OPTION);
            KeypointDef def = new KeypointDef(name.trim().replaceAll("\\s+", "_"),
                    Color.getHSBColor((float) Math.random(), 0.75f, 1.0f),
                    isPawOpt == JOptionPane.YES_OPTION);
            skeletonDefs.add(def);
            kpListModel.addElement(def);
        }
    }

    // --- Modern Interaktív Videó Vászon (Drag & Drop + Jobb klikk törlés + Képaránytartás) ---
    private class VideoCanvas extends JPanel {
        private static final double HIT_RADIUS_PX = 14.0;
        private String hoveredKeypoint = null;
        private String draggedKeypoint = null;
        private int hoveredLine = 0; // 1=Ladder, 2=Fault, 3=Fall
        private int draggedLine = 0;
        private int hoveredRungIdx = -1;
        private int draggedRungIdx = -1;

        public VideoCanvas() {
            setBackground(new Color(22, 24, 28));
            setFocusable(true);

            MouseAdapter ma = new MouseAdapter() {
                @Override
                public void mouseMoved(MouseEvent e) {
                    if (currentImage == null) return;
                    updateHoverState(e.getPoint());
                }

                @Override
                public void mousePressed(MouseEvent e) {
                    requestFocusInWindow();
                    if (currentImage == null) return;
                    updateHoverState(e.getPoint());

                    Point2D imgPt = screenToImage(e.getPoint());
                    if (imgPt == null) return;

                    // JOBB KLIKK: Pont vagy Létrafok törlése
                    if (SwingUtilities.isRightMouseButton(e)) {
                        if (hoveredKeypoint != null) {
                            FrameAnnotation fa = annotations.get(currentFrameIdx);
                            if (fa != null) {
                                fa.keypoints.remove(hoveredKeypoint);
                                if (fa.keypoints.isEmpty()) annotations.remove(currentFrameIdx);
                                else updateBoundingBox(fa);
                                statusLabel.setText(" Törölve: " + hoveredKeypoint + " (Frame " + currentFrameIdx + ")");
                                kpList.repaint();
                                updateHoverState(e.getPoint());
                                repaint();
                            }
                        } else if (hoveredRungIdx != -1) {
                            ladderRungsX.remove(hoveredRungIdx);
                            hoveredRungIdx = -1;
                            repaint();
                        }
                        return;
                    }

                    // BAL KLIKK: Meglévő pont, létrafok vagy vonal megfogása (Drag)
                    if (SwingUtilities.isLeftMouseButton(e)) {
                        if (hoveredKeypoint != null) {
                            draggedKeypoint = hoveredKeypoint;
                            selectKeypointInList(draggedKeypoint);
                            return;
                        }
                        if (hoveredRungIdx != -1) {
                            draggedRungIdx = hoveredRungIdx;
                            return;
                        }
                        if (hoveredLine != 0 && chkShowLines.isSelected()) {
                            draggedLine = hoveredLine;
                            return;
                        }

                        // Ha Létrafok lerakási mód van bekapcsolva
                        if (chkAddRungMode.isSelected()) {
                            ladderRungsX.add(imgPt.x);
                            draggedRungIdx = ladderRungsX.size() - 1;
                            repaint();
                            return;
                        }

                        // Üres helyre kattintás: Aktuálisan kijelölt testrész lerakása és azonnali mozgathatósága
                        KeypointDef selectedKp = kpList.getSelectedValue();
                        if (selectedKp != null) {
                            FrameAnnotation fa = annotations.computeIfAbsent(currentFrameIdx, k -> {
                                FrameAnnotation n = new FrameAnnotation();
                                n.frameIndex = k;
                                return n;
                            });
                            fa.keypoints.put(selectedKp.name, new Point2D(imgPt.x, imgPt.y, 1.0));
                            updateBoundingBox(fa);
                            draggedKeypoint = selectedKp.name;
                            hoveredKeypoint = selectedKp.name;
                            kpList.repaint();
                            repaint();
                        }
                    }
                }

                @Override
                public void mouseDragged(MouseEvent e) {
                    if (currentImage == null) return;
                    Point2D imgPt = screenToImageClamped(e.getPoint());

                    if (draggedKeypoint != null) {
                        FrameAnnotation fa = annotations.get(currentFrameIdx);
                        if (fa != null && fa.keypoints.containsKey(draggedKeypoint)) {
                            Point2D pt = fa.keypoints.get(draggedKeypoint);
                            pt.x = imgPt.x;
                            pt.y = imgPt.y;
                            updateBoundingBox(fa);
                            repaint();
                        }
                    } else if (draggedRungIdx != -1 && draggedRungIdx < ladderRungsX.size()) {
                        ladderRungsX.set(draggedRungIdx, imgPt.x);
                        repaint();
                    } else if (draggedLine != 0) {
                        if (draggedLine == 1) ladderLevelY = imgPt.y;
                        else if (draggedLine == 2) faultThresholdY = imgPt.y;
                        else if (draggedLine == 3) fallThresholdY = imgPt.y;
                        repaint();
                    }
                }

                @Override
                public void mouseReleased(MouseEvent e) {
                    if (draggedKeypoint != null) {
                        // Automatikus léptetés a következő még le nem rakott testrészre
                        advanceToNextUnplacedKeypoint();
                    }
                    draggedKeypoint = null;
                    draggedLine = 0;
                    draggedRungIdx = -1;
                    updateHoverState(e.getPoint());
                }
            };

            addMouseListener(ma);
            addMouseMotionListener(ma);
        }

        private void selectKeypointInList(String name) {
            for (int i = 0; i < kpListModel.size(); i++) {
                if (kpListModel.get(i).name.equals(name)) {
                    kpList.setSelectedIndex(i);
                    break;
                }
            }
        }

        private void advanceToNextUnplacedKeypoint() {
            FrameAnnotation fa = annotations.get(currentFrameIdx);
            if (fa == null) return;
            int start = kpList.getSelectedIndex();
            for (int offset = 1; offset <= kpListModel.size(); offset++) {
                int idx = (start + offset) % kpListModel.size();
                if (!fa.keypoints.containsKey(kpListModel.get(idx).name)) {
                    kpList.setSelectedIndex(idx);
                    return;
                }
            }
        }

        private Rectangle getVideoBounds() {
            if (currentImage == null) return new Rectangle(0, 0, getWidth(), getHeight());
            double imgAspect = (double) currentImage.getWidth() / currentImage.getHeight();
            double panelAspect = (double) getWidth() / getHeight();
            int drawW, drawH, drawX, drawY;
            if (panelAspect > imgAspect) {
                drawH = getHeight();
                drawW = (int) Math.round(drawH * imgAspect);
                drawX = (getWidth() - drawW) / 2;
                drawY = 0;
            } else {
                drawW = getWidth();
                drawH = (int) Math.round(drawW / imgAspect);
                drawX = 0;
                drawY = (getHeight() - drawH) / 2;
            }
            return new Rectangle(drawX, drawY, Math.max(1, drawW), Math.max(1, drawH));
        }

        private Point2D screenToImage(Point p) {
            Rectangle vb = getVideoBounds();
            if (!vb.contains(p)) return null;
            double ix = (p.x - vb.x) * ((double) currentImage.getWidth() / vb.width);
            double iy = (p.y - vb.y) * ((double) currentImage.getHeight() / vb.height);
            return new Point2D(ix, iy, 1.0);
        }

        private Point2D screenToImageClamped(Point p) {
            Rectangle vb = getVideoBounds();
            double clampedX = Math.max(vb.x, Math.min(vb.x + vb.width, p.x));
            double clampedY = Math.max(vb.y, Math.min(vb.y + vb.height, p.y));
            double ix = (clampedX - vb.x) * ((double) currentImage.getWidth() / vb.width);
            double iy = (clampedY - vb.y) * ((double) currentImage.getHeight() / vb.height);
            return new Point2D(ix, iy, 1.0);
        }

        private Point imageToScreen(double ix, double iy, Rectangle vb) {
            int sx = vb.x + (int) Math.round(ix * ((double) vb.width / currentImage.getWidth()));
            int sy = vb.y + (int) Math.round(iy * ((double) vb.height / currentImage.getHeight()));
            return new Point(sx, sy);
        }

        private void updateHoverState(Point mousePt) {
            String prevHoverKp = hoveredKeypoint;
            int prevHoverLine = hoveredLine;
            int prevHoverRung = hoveredRungIdx;

            hoveredKeypoint = null;
            hoveredLine = 0;
            hoveredRungIdx = -1;

            Rectangle vb = getVideoBounds();
            FrameAnnotation fa = annotations.get(currentFrameIdx);

            // 1. Kulcspontok vizsgálata
            if (fa != null) {
                double bestDist = HIT_RADIUS_PX;
                for (Map.Entry<String, Point2D> entry : fa.keypoints.entrySet()) {
                    Point sp = imageToScreen(entry.getValue().x, entry.getValue().y, vb);
                    double d = sp.distance(mousePt);
                    if (d <= bestDist) {
                        bestDist = d;
                        hoveredKeypoint = entry.getKey();
                    }
                }
            }

            // 2. Létrafokok vizsgálata
            if (hoveredKeypoint == null) {
                for (int i = 0; i < ladderRungsX.size(); i++) {
                    Point sp = imageToScreen(ladderRungsX.get(i), ladderLevelY, vb);
                    if (sp.distance(mousePt) <= HIT_RADIUS_PX) {
                        hoveredRungIdx = i;
                        break;
                    }
                }
            }

            // 3. Vízszintes határvonalak vizsgálata
            if (hoveredKeypoint == null && hoveredRungIdx == -1 && chkShowLines.isSelected() && vb.contains(mousePt)) {
                int yLadder = imageToScreen(0, ladderLevelY, vb).y;
                int yFault = imageToScreen(0, faultThresholdY, vb).y;
                int yFall = imageToScreen(0, fallThresholdY, vb).y;

                if (Math.abs(mousePt.y - yLadder) <= 7) hoveredLine = 1;
                else if (Math.abs(mousePt.y - yFault) <= 7) hoveredLine = 2;
                else if (Math.abs(mousePt.y - yFall) <= 7) hoveredLine = 3;
            }

            // Kurzor beállítása
            if (hoveredKeypoint != null || hoveredRungIdx != -1) {
                setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
            } else if (hoveredLine != 0) {
                setCursor(Cursor.getPredefinedCursor(Cursor.N_RESIZE_CURSOR));
            } else {
                setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
            }

            if (!Objects.equals(prevHoverKp, hoveredKeypoint) || prevHoverLine != hoveredLine || prevHoverRung != hoveredRungIdx) {
                repaint();
            }
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);

            if (currentImage == null) {
                g2.setColor(new Color(130, 140, 155));
                g2.setFont(getFont().deriveFont(Font.BOLD, 16f));
                String msg = "Nyiss meg egy videót az 1-es gombbal!";
                int sw = g2.getFontMetrics().stringWidth(msg);
                g2.drawString(msg, (getWidth() - sw) / 2, getHeight() / 2);
                return;
            }

            Rectangle vb = getVideoBounds();
            g2.drawImage(currentImage, vb.x, vb.y, vb.width, vb.height, null);

            // 1. Vízszintes határvonalak kirajzolása
            if (chkShowLines.isSelected()) {
                drawInteractiveLine(g2, vb, ladderLevelY, new Color(46, 213, 115), "Létrafok síkja (Húzd az egérrel)", hoveredLine == 1 || draggedLine == 1);
                drawInteractiveLine(g2, vb, faultThresholdY, new Color(255, 165, 2), "Mellélépés határ (Húzd az egérrel)", hoveredLine == 2 || draggedLine == 2);
                drawInteractiveLine(g2, vb, fallThresholdY, new Color(255, 71, 87), "Elesés határ (Húzd az egérrel)", hoveredLine == 3 || draggedLine == 3);
            }

            // 2. Létrafokok kirajzolása
            for (int i = 0; i < ladderRungsX.size(); i++) {
                Point sp = imageToScreen(ladderRungsX.get(i), ladderLevelY, vb);
                boolean active = (i == hoveredRungIdx || i == draggedRungIdx);
                g2.setColor(active ? Color.WHITE : new Color(46, 213, 115));
                g2.fill(new RoundRectangle2D.Double(sp.x - 5, sp.y - 8, 10, 16, 4, 4));
                g2.setColor(Color.BLACK);
                g2.draw(new RoundRectangle2D.Double(sp.x - 5, sp.y - 8, 10, 16, 4, 4));
            }

            // 3. Csontváz és Kulcspontok kirajzolása
            FrameAnnotation fa = annotations.get(currentFrameIdx);
            if (fa != null) {
                g2.setStroke(new BasicStroke(2.5f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
                for (int[] bone : skeletonBones) {
                    if (bone[0] < skeletonDefs.size() && bone[1] < skeletonDefs.size()) {
                        Point2D p1 = fa.keypoints.get(skeletonDefs.get(bone[0]).name);
                        Point2D p2 = fa.keypoints.get(skeletonDefs.get(bone[1]).name);
                        if (p1 != null && p2 != null) {
                            Point s1 = imageToScreen(p1.x, p1.y, vb);
                            Point s2 = imageToScreen(p2.x, p2.y, vb);
                            g2.setColor(new Color(255, 255, 255, 190));
                            g2.draw(new Line2D.Double(s1.x, s1.y, s2.x, s2.y));
                        }
                    }
                }

                for (KeypointDef def : skeletonDefs) {
                    Point2D pt = fa.keypoints.get(def.name);
                    if (pt != null) {
                        Point sp = imageToScreen(pt.x, pt.y, vb);
                        boolean isHovered = def.name.equals(hoveredKeypoint) || def.name.equals(draggedKeypoint);
                        int r = isHovered ? 9 : 6;

                        if (isHovered) {
                            g2.setColor(new Color(255, 255, 255, 90));
                            g2.fill(new Ellipse2D.Double(sp.x - r - 4, sp.y - r - 4, (r + 4) * 2, (r + 4) * 2));
                        }

                        g2.setColor(def.color);
                        g2.fill(new Ellipse2D.Double(sp.x - r, sp.y - r, r * 2, r * 2));
                        g2.setStroke(new BasicStroke(isHovered ? 2.5f : 1.5f));
                        g2.setColor(isHovered ? Color.WHITE : Color.BLACK);
                        g2.draw(new Ellipse2D.Double(sp.x - r, sp.y - r, r * 2, r * 2));

                        // Felirat háttérrel a jó olvashatóságért
                        String label = def.name;
                        FontMetrics fm = g2.getFontMetrics();
                        int tw = fm.stringWidth(label);
                        g2.setColor(new Color(0, 0, 0, 165));
                        g2.fillRoundRect(sp.x + 10, sp.y - 12, tw + 8, 18, 6, 6);
                        g2.setColor(isHovered ? Color.YELLOW : Color.WHITE);
                        g2.drawString(label, sp.x + 14, sp.y + 1);
                    }
                }
            }
        }

        private void drawInteractiveLine(Graphics2D g2, Rectangle vb, double imgY, Color color, String label, boolean highlighted) {
            int sy = imageToScreen(0, imgY, vb).y;
            g2.setColor(color);
            float width = highlighted ? 3.0f : 1.8f;
            g2.setStroke(new BasicStroke(width, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0, new float[]{10, 6}, 0));
            g2.drawLine(vb.x, sy, vb.x + vb.width, sy);

            FontMetrics fm = g2.getFontMetrics();
            int tw = fm.stringWidth(label);
            g2.fillRoundRect(vb.x + 8, sy - 20, tw + 12, 18, 6, 6);
            g2.setColor(Color.BLACK);
            g2.drawString(label, vb.x + 14, sy - 7);
        }
    }

    // Színes pont ikon a jobb oldali listához
    private static class ColorDotIcon implements Icon {
        private final Color color;
        private final boolean filled;

        public ColorDotIcon(Color color, boolean filled) {
            this.color = color;
            this.filled = filled;
        }
        @Override public void paintIcon(Component c, Graphics g, int x, int y) {
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setColor(color);
            g2.fillOval(x + 2, y + 2, 12, 12);
            if (filled) {
                g2.setColor(Color.WHITE);
                g2.drawOval(x + 1, y + 1, 14, 14);
            }
            g2.dispose();
        }
        @Override public int getIconWidth() { return 18; }
        @Override public int getIconHeight() { return 16; }
    }

    private void updateBoundingBox(FrameAnnotation fa) {
        double minX = Double.MAX_VALUE, minY = Double.MAX_VALUE;
        double maxX = 0, maxY = 0;
        for (Point2D pt : fa.keypoints.values()) {
            minX = Math.min(minX, pt.x);
            minY = Math.min(minY, pt.y);
            maxX = Math.max(maxX, pt.x);
            maxY = Math.max(maxY, pt.y);
        }
        int pad = 25;
        fa.boundingBox = new Rectangle((int) Math.max(0, minX - pad), (int) Math.max(0, minY - pad),
                (int) (maxX - minX + 2 * pad), (int) (maxY - minY + 2 * pad));
    }

    // --- 2. LÉPÉS: YOLO-Pose Tanító Adathalmaz Mentése (Háttérszálon ProgressBarral) ---
    private void exportYoloDataset() {
        if (annotations.isEmpty() || currentVideoFile == null) {
            JOptionPane.showMessageDialog(this, "Előbb jelölj be kulcspontokat a videón!");
            return;
        }

        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        chooser.setDialogTitle("Válassz mappát a YOLO tanító adathalmaznak");
        if (chooser.showSaveDialog(this) != JFileChooser.APPROVE_OPTION) return;

        Path rootDir = chooser.getSelectedFile().toPath();
        progressBar.setValue(0);
        progressBar.setVisible(true);
        statusLabel.setText(" Tanító adathalmaz mentése folyamatban...");

        new SwingWorker<Void, Integer>() {
            @Override
            protected Void doInBackground() throws Exception {
                Path imgDir = rootDir.resolve("images/train");
                Path lblDir = rootDir.resolve("labels/train");
                Files.createDirectories(imgDir);
                Files.createDirectories(lblDir);

                int imgW = currentImage.getWidth();
                int imgH = currentImage.getHeight();
                int count = 0;
                int total = annotations.size();

                for (Map.Entry<Integer, FrameAnnotation> entry : annotations.entrySet()) {
                    int fIdx = entry.getKey();
                    FrameAnnotation fa = entry.getValue();
                    updateBoundingBox(fa);

                    synchronized (MouseLadderTracker.this) {
                        grabber.setVideoFrameNumber(fIdx);
                        BufferedImage bi = converter.getBufferedImage(grabber.grabImage());
                        String baseName = String.format("frame_%05d", fIdx);
                        ImageIO.write(bi, "jpg", imgDir.resolve(baseName + ".jpg").toFile());

                        StringBuilder sb = new StringBuilder("0 ");
                        double cx = (fa.boundingBox.x + fa.boundingBox.width / 2.0) / imgW;
                        double cy = (fa.boundingBox.y + fa.boundingBox.height / 2.0) / imgH;
                        double bw = (double) fa.boundingBox.width / imgW;
                        double bh = (double) fa.boundingBox.height / imgH;
                        sb.append(String.format(Locale.US, "%.6f %.6f %.6f %.6f",
                                clamp(cx), clamp(cy), clamp(bw), clamp(bh)));

                        for (KeypointDef def : skeletonDefs) {
                            Point2D pt = fa.keypoints.get(def.name);
                            if (pt != null) {
                                sb.append(String.format(Locale.US, " %.6f %.6f 2",
                                        clamp(pt.x / imgW), clamp(pt.y / imgH)));
                            } else {
                                sb.append(" 0.000000 0.000000 0");
                            }
                        }
                        Files.writeString(lblDir.resolve(baseName + ".txt"), sb.toString() + "\n");
                    }
                    count++;
                    publish((count * 100) / total);
                }

                String yaml = String.format(
                        "path: %s\ntrain: images/train\nval: images/train\n\n" +
                        "kpt_shape: [%d, 3]\nnames:\n  0: mouse\n",
                        rootDir.toAbsolutePath().toString().replace("\\", "/"),
                        skeletonDefs.size()
                );
                Files.writeString(rootDir.resolve("mouse_pose.yaml"), yaml);

                String pyScript =
                        "from ultralytics import YOLO\n\n" +
                        "if __name__ == '__main__':\n" +
                        "    model = YOLO('yolo11n-pose.pt')\n" +
                        "    model.train(data='mouse_pose.yaml', epochs=120, imgsz=640, batch=8)\n" +
                        "    model.export(format='onnx', imgsz=640)\n" +
                        "    print('KÉSZ! A best.onnx modell betölthető a Java programba.')\n";
                Files.writeString(rootDir.resolve("train_and_export_onnx.py"), pyScript);
                return null;
            }

            @Override
            protected void process(List<Integer> chunks) {
                progressBar.setValue(chunks.get(chunks.size() - 1));
            }

            @Override
            protected void done() {
                progressBar.setVisible(false);
                seekToFrame(currentFrameIdx);
                statusLabel.setText(" Tanító adathalmaz sikeresen mentve: " + rootDir);
                JOptionPane.showMessageDialog(MouseLadderTracker.this,
                        "Adathalmaz mentve ide:\n" + rootDir + "\n\nTanítás indítása PowerShellben:\n" +
                        ".\\venv\\Scripts\\python.exe train_and_export_onnx.py");
            }
        }.execute();
    }

    private double clamp(double v) { return Math.max(0.0, Math.min(1.0, v)); }

    // --- 3. LÉPÉS: ONNX AI Modell betöltése ---
    private void loadOnnxModel() {
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("Válaszd ki a betanított best.onnx modellt");
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                if (ortEnv == null) ortEnv = OrtEnvironment.getEnvironment();
                if (ortSession != null) ortSession.close();
                ortSession = ortEnv.createSession(chooser.getSelectedFile().getAbsolutePath(), new OrtSession.SessionOptions());
                statusLabel.setText(" AI Modell aktív: " + chooser.getSelectedFile().getName());
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this, "ONNX betöltési hiba: " + ex.getMessage());
            }
        }
    }

    private Map<String, Point2D> runOnnxInference(BufferedImage img) throws Exception {
        int inputSize = 640;
        BufferedImage resized = new BufferedImage(inputSize, inputSize, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.drawImage(img, 0, 0, inputSize, inputSize, null);
        g.dispose();

        float[] chwData = new float[3 * inputSize * inputSize];
        int area = inputSize * inputSize;
        for (int y = 0; y < inputSize; y++) {
            for (int x = 0; x < inputSize; x++) {
                int rgb = resized.getRGB(x, y);
                int idx = y * inputSize + x;
                chwData[idx] = ((rgb >> 16) & 0xFF) / 255.0f;
                chwData[area + idx] = ((rgb >> 8) & 0xFF) / 255.0f;
                chwData[2 * area + idx] = (rgb & 0xFF) / 255.0f;
            }
        }

        String inputName = ortSession.getInputNames().iterator().next();
        OnnxTensor tensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(chwData), new long[]{1, 3, inputSize, inputSize});

        try (OrtSession.Result result = ortSession.run(Collections.singletonMap(inputName, tensor))) {
            float[][][] output = (float[][][]) result.get(0).getValue();
            int numAnchors = output[0][0].length;
            int bestAnchor = -1;
            float maxScore = 0.25f;

            for (int i = 0; i < numAnchors; i++) {
                float score = output[0][4][i];
                if (score > maxScore) {
                    maxScore = score;
                    bestAnchor = i;
                }
            }

            Map<String, Point2D> detected = new LinkedHashMap<>();
            if (bestAnchor != -1) {
                double scaleX = (double) img.getWidth() / inputSize;
                double scaleY = (double) img.getHeight() / inputSize;
                for (int k = 0; k < skeletonDefs.size(); k++) {
                    int baseIdx = 5 + k * 3;
                    if (baseIdx + 2 < output[0].length) {
                        double kx = output[0][baseIdx][bestAnchor] * scaleX;
                        double ky = output[0][baseIdx + 1][bestAnchor] * scaleY;
                        double kconf = output[0][baseIdx + 2][bestAnchor];
                        detected.put(skeletonDefs.get(k).name, new Point2D(kx, ky, kconf));
                    }
                }
            }
            return detected;
        } finally {
            tensor.close();
        }
    }

    // --- 4. LÉPÉS: Gyors Szekvenciális Videóelemzés és CSV Export (Háttérszálon) ---
    private void runFullVideoAnalysis() {
        if (grabber == null) {
            JOptionPane.showMessageDialog(this, "Előbb nyiss meg egy videót!");
            return;
        }
        if (playTimer.isRunning()) togglePlayPause();

        try {
            ladderLengthCm = Double.parseDouble(txtLadderCm.getText().trim());
        } catch (NumberFormatException ignored) {}

        JFileChooser chooser = new JFileChooser();
        chooser.setSelectedFile(new File("egermozgas_elemzes.csv"));
        if (chooser.showSaveDialog(this) != JFileChooser.APPROVE_OPTION) return;

        File csvFile = chooser.getSelectedFile();
        File summaryFile = new File(csvFile.getParentFile(), "egermozgas_osszesito_statisztika.csv");

        progressBar.setValue(0);
        progressBar.setVisible(true);
        statusLabel.setText(" Videó elemzése folyamatban...");

        new SwingWorker<String, Integer>() {
            @Override
            protected String doInBackground() throws Exception {
                try (PrintWriter rawOut = new PrintWriter(new OutputStreamWriter(new FileOutputStream(csvFile), StandardCharsets.UTF_8));
                     PrintWriter sumOut = new PrintWriter(new OutputStreamWriter(new FileOutputStream(summaryFile), StandardCharsets.UTF_8))) {

                    StringBuilder header = new StringBuilder("Frame;Time_s");
                    for (KeypointDef def : skeletonDefs) {
                        header.append(String.format(";%s_X_px;%s_Y_px;%s_Conf", def.name, def.name, def.name));
                    }
                    header.append(";Body_Speed_cm_s;Active_Step_Event;Fault_Type\n");
                    rawOut.write(header.toString());

                    int totalSteps = 0, totalSlips = 0, totalFalls = 0;
                    boolean inFallState = false;
                    Map<String, Boolean> pawInSwingPhase = new HashMap<>();
                    Map<String, Boolean> pawInSlipState = new HashMap<>();
                    Map<String, Point2D> prevPoints = new HashMap<>();

                    double pxToCm = ladderLengthCm / Math.max(1.0, currentImage.getWidth());
                    double dt = 1.0 / videoFps;
                    double speedSum = 0.0;
                    int validSpeedFrames = 0;

                    synchronized (MouseLadderTracker.this) {
                        grabber.setVideoFrameNumber(0);
                    }

                    for (int f = 0; f < totalFrames; f++) {
                        BufferedImage img = null;
                        if (ortSession != null) {
                            synchronized (MouseLadderTracker.this) {
                                Frame frame = grabber.grabImage();
                                if (frame != null) img = converter.getBufferedImage(frame);
                            }
                        }

                        Map<String, Point2D> currentPts;
                        if (ortSession != null && img != null) {
                            currentPts = runOnnxInference(img);
                            FrameAnnotation fa = new FrameAnnotation();
                            fa.frameIndex = f;
                            fa.keypoints.putAll(currentPts);
                            annotations.put(f, fa);
                        } else {
                            FrameAnnotation fa = annotations.get(f);
                            currentPts = (fa != null) ? fa.keypoints : Collections.emptyMap();
                        }

                        String refNode = currentPts.containsKey("Gerinc_Kozep") ? "Gerinc_Kozep" : skeletonDefs.get(0).name;
                        double instantSpeedCmS = 0.0;
                        if (currentPts.containsKey(refNode) && prevPoints.containsKey(refNode)) {
                            Point2D pNow = currentPts.get(refNode);
                            Point2D pPrev = prevPoints.get(refNode);
                            double distPx = Math.hypot(pNow.x - pPrev.x, pNow.y - pPrev.y);
                            instantSpeedCmS = (distPx * pxToCm) / dt;
                            speedSum += instantSpeedCmS;
                            validSpeedFrames++;
                        }

                        List<String> stepEvents = new ArrayList<>();
                        List<String> faultEvents = new ArrayList<>();
                        int pawsBelowFallLine = 0;

                        for (KeypointDef def : skeletonDefs) {
                            if (!def.isPaw) continue;
                            Point2D pNow = currentPts.get(def.name);
                            if (pNow == null) continue;

                            Point2D pPrev = prevPoints.get(def.name);
                            if (pPrev != null) {
                                double vxCmS = (Math.abs(pNow.x - pPrev.x) * pxToCm) / dt;
                                boolean wasSwinging = pawInSwingPhase.getOrDefault(def.name, false);
                                if (!wasSwinging && vxCmS > 8.0) {
                                    pawInSwingPhase.put(def.name, true);
                                } else if (wasSwinging && vxCmS < 2.5 && pNow.y <= faultThresholdY) {
                                    pawInSwingPhase.put(def.name, false);
                                    totalSteps++;
                                    stepEvents.add("LEPES_" + def.name);
                                }
                            }

                            boolean wasSlipping = pawInSlipState.getOrDefault(def.name, false);
                            if (pNow.y > faultThresholdY) {
                                if (!wasSlipping) {
                                    pawInSlipState.put(def.name, true);
                                    totalSlips++;
                                    faultEvents.add("MELLÉLÉPÉS_" + def.name);
                                }
                            } else {
                                pawInSlipState.put(def.name, false);
                            }

                            if (pNow.y > fallThresholdY) pawsBelowFallLine++;
                        }

                        Point2D spinePt = currentPts.get(refNode);
                        boolean fallCondition = (spinePt != null && spinePt.y > faultThresholdY) || (pawsBelowFallLine >= 2);
                        if (fallCondition) {
                            if (!inFallState) {
                                inFallState = true;
                                totalFalls++;
                                faultEvents.add("ELESES");
                            }
                        } else {
                            inFallState = false;
                        }

                        StringBuilder row = new StringBuilder(String.format(Locale.US, "%d;%.4f", f, f * dt));
                        for (KeypointDef def : skeletonDefs) {
                            Point2D pt = currentPts.get(def.name);
                            if (pt != null) {
                                row.append(String.format(Locale.US, ";%.2f;%.2f;%.2f", pt.x, pt.y, pt.conf));
                            } else {
                                row.append(";;;");
                            }
                        }
                        row.append(String.format(Locale.US, ";%.3f;%s;%s\n",
                                instantSpeedCmS,
                                stepEvents.isEmpty() ? "NONE" : String.join("|", stepEvents),
                                faultEvents.isEmpty() ? "OK" : String.join("|", faultEvents)));
                        rawOut.write(row.toString());

                        prevPoints.putAll(currentPts);
                        publish((f * 100) / Math.max(1, totalFrames));
                    }

                    double avgSpeed = validSpeedFrames > 0 ? speedSum / validSpeedFrames : 0.0;
                    double totalTimeS = totalFrames / videoFps;
                    sumOut.write("Video_File;Total_Duration_s;Avg_Crossing_Speed_cm_s;Total_Steps;Foot_Slips_Misses;Falls;Success_Ratio_Percent\n");
                    double successRatio = totalSteps > 0 ? Math.max(0.0, 100.0 * (totalSteps - totalSlips) / totalSteps) : 100.0;
                    sumOut.write(String.format(Locale.US, "%s;%.2f;%.2f;%d;%d;%d;%.2f\n",
                            currentVideoFile.getName(), totalTimeS, avgSpeed, totalSteps, totalSlips, totalFalls, successRatio));

                    return String.format(
                            "Elemzés kész!\n\nÁtlagsebesség: %.2f cm/s\nÖsszes lépésszám: %d\n" +
                            "Mellélépések: %d\nElesések: %d\n\nRészletes CSV: %s\nÖsszesítő CSV: %s",
                            avgSpeed, totalSteps, totalSlips, totalFalls, csvFile.getName(), summaryFile.getName());
                }
            }

            @Override
            protected void process(List<Integer> chunks) {
                progressBar.setValue(chunks.get(chunks.size() - 1));
            }

            @Override
            protected void done() {
                progressBar.setVisible(false);
                seekToFrame(currentFrameIdx);
                try {
                    String msg = get();
                    statusLabel.setText(" Elemzés sikeresen befejezve és CSV-be mentve.");
                    JOptionPane.showMessageDialog(MouseLadderTracker.this, msg);
                } catch (Exception ex) {
                    ex.printStackTrace();
                    JOptionPane.showMessageDialog(MouseLadderTracker.this, "Hiba az elemzés során: " + ex.getMessage());
                }
            }
        }.execute();
    }

    public static void main(String[] args) {
        // FFmpeg HEVC / Dolby Vision figyelmeztetések kikapcsolása
        avutil.av_log_set_level(avutil.AV_LOG_ERROR);

        // Modern FlatLaf sötét téma beállítása lekerekített elemekkel
        FlatDarkLaf.setup();
        UIManager.put("Button.arc", 10);
        UIManager.put("Component.arc", 10);
        UIManager.put("ProgressBar.arc", 10);

        SwingUtilities.invokeLater(() -> new MouseLadderTracker().setVisible(true));
    }
}