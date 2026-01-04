import {useCurrentFrame, useVideoConfig} from 'remotion';

interface Pose {
  rightShoulder: number;
  rightElbow: number;
  leftShoulder: number;
  leftElbow: number;
  rightHip: number;
  rightKnee: number;
  leftHip: number;
  leftKnee: number;
}

export const POSES: Record<string, Pose> = {
  standing: { rightShoulder: 15, rightElbow: 20, leftShoulder: 15, leftElbow: 20, rightHip: 5, rightKnee: 0, leftHip: 5, leftKnee: 0 },
  waving: { rightShoulder: 150, rightElbow: 35, leftShoulder: 15, leftElbow: 20, rightHip: 5, rightKnee: 0, leftHip: 5, leftKnee: 0 },
  pointing: { rightShoulder: 90, rightElbow: 0, leftShoulder: 15, leftElbow: 20, rightHip: 5, rightKnee: 0, leftHip: 5, leftKnee: 0 },
  presenting: { rightShoulder: 120, rightElbow: 10, leftShoulder: 15, leftElbow: 20, rightHip: 5, rightKnee: 0, leftHip: 5, leftKnee: 0 },
  thinking: { rightShoulder: 110, rightElbow: 140, leftShoulder: 15, leftElbow: 20, rightHip: 5, rightKnee: 0, leftHip: 5, leftKnee: 0 },
  celebrating: { rightShoulder: 160, rightElbow: 20, leftShoulder: 160, leftElbow: 20, rightHip: 5, rightKnee: 0, leftHip: 5, leftKnee: 0 },
  jumping: { rightShoulder: 140, rightElbow: 10, leftShoulder: 140, leftElbow: 10, rightHip: 25, rightKnee: 35, leftHip: 25, leftKnee: 35 },
  walking1: { rightShoulder: 30, rightElbow: 40, leftShoulder: 5, leftElbow: 10, rightHip: -10, rightKnee: 15, leftHip: 20, leftKnee: 30 },
  walking2: { rightShoulder: 5, rightElbow: 10, leftShoulder: 30, leftElbow: 40, rightHip: 20, rightKnee: 30, leftHip: -10, leftKnee: 15 },
};

function lerp(a: number, b: number, t: number) { return a + (b - a) * t; }

function interpolatePose(from: Pose, to: Pose, t: number): Pose {
  return {
    rightShoulder: lerp(from.rightShoulder, to.rightShoulder, t),
    rightElbow: lerp(from.rightElbow, to.rightElbow, t),
    leftShoulder: lerp(from.leftShoulder, to.leftShoulder, t),
    leftElbow: lerp(from.leftElbow, to.leftElbow, t),
    rightHip: lerp(from.rightHip, to.rightHip, t),
    rightKnee: lerp(from.rightKnee, to.rightKnee, t),
    leftHip: lerp(from.leftHip, to.leftHip, t),
    leftKnee: lerp(from.leftKnee, to.leftKnee, t),
  };
}

function pt(x: number, y: number, angleDeg: number, dist: number) {
  const rad = angleDeg * Math.PI / 180;
  return { x: x + Math.sin(rad) * dist, y: y + Math.cos(rad) * dist };
}

// Simple limb as rotated rounded rectangle
function limbPath(start: {x: number, y: number}, end: {x: number, y: number}, width: number): string {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len < 1) return '';

  // Perpendicular offset
  const px = (-dy / len) * width / 2;
  const py = (dx / len) * width / 2;

  const r = width / 2;

  return `
    M ${start.x + px} ${start.y + py}
    L ${end.x + px} ${end.y + py}
    A ${r} ${r} 0 0 1 ${end.x - px} ${end.y - py}
    L ${start.x - px} ${start.y - py}
    A ${r} ${r} 0 0 1 ${start.x + px} ${start.y + py}
    Z
  `;
}

interface Props {
  pose: Pose;
  x?: number;
  y?: number;
  scale?: number;
}

export const UnifiedBoneCharacter: React.FC<Props> = ({ pose, x = 960, y = 540, scale = 1 }) => {
  // Dimensions
  const headR = 100;
  const bodyW = 110;
  const bodyH = 200;
  const bodyY = headR + 10; // Body starts just below head

  const shoulderX = 45;
  const shoulderY = bodyY + 30;
  const hipX = 35;
  const hipY = bodyY + bodyH - 30;

  const upperArmLen = 85;
  const lowerArmLen = 80;
  const armW = 42;
  const handR = 28;

  const upperLegLen = 100;
  const lowerLegLen = 95;
  const legW = 48;
  const footR = 26;

  // Joint positions - RIGHT SIDE
  const rShoulder = { x: shoulderX, y: shoulderY };
  const rElbow = pt(rShoulder.x, rShoulder.y, pose.rightShoulder, upperArmLen);
  const rWrist = pt(rElbow.x, rElbow.y, pose.rightShoulder + pose.rightElbow, lowerArmLen);

  const rHip = { x: hipX, y: hipY };
  const rKnee = pt(rHip.x, rHip.y, pose.rightHip, upperLegLen);
  const rAnkle = pt(rKnee.x, rKnee.y, pose.rightHip - pose.rightKnee, lowerLegLen);

  // Joint positions - LEFT SIDE (mirrored angles)
  const lShoulder = { x: -shoulderX, y: shoulderY };
  const lElbow = pt(lShoulder.x, lShoulder.y, -pose.leftShoulder, upperArmLen);
  const lWrist = pt(lElbow.x, lElbow.y, -pose.leftShoulder - pose.leftElbow, lowerArmLen);

  const lHip = { x: -hipX, y: hipY };
  const lKnee = pt(lHip.x, lHip.y, -pose.leftHip, upperLegLen);
  const lAnkle = pt(lKnee.x, lKnee.y, -pose.leftHip + pose.leftKnee, lowerLegLen);

  // Simple oval body
  const bodyPath = `
    M 0 ${bodyY}
    C ${bodyW} ${bodyY} ${bodyW} ${bodyY + bodyH} 0 ${bodyY + bodyH}
    C ${-bodyW} ${bodyY + bodyH} ${-bodyW} ${bodyY} 0 ${bodyY}
    Z
  `;

  const fill = "url(#grad)";
  const stroke = "#1a1a2e";
  const sw = 9;

  return (
    <svg width={1920} height={1080} style={{ position: 'absolute', left: 0, top: 0 }}>
      <defs>
        <radialGradient id="grad" cx="50%" cy="30%" r="70%">
          <stop offset="0%" stopColor="#FFFFFF"/>
          <stop offset="100%" stopColor="#E8E8F0"/>
        </radialGradient>
        <filter id="shad" x="-50%" y="-50%" width="200%" height="200%">
          <feDropShadow dx="5" dy="8" stdDeviation="4" floodColor="#1a1a2e" floodOpacity="0.25"/>
        </filter>
      </defs>

      <g transform={`translate(${x}, ${y}) scale(${scale})`} filter="url(#shad)">
        {/* Ground shadow */}
        <ellipse cx={0} cy={hipY + upperLegLen + lowerLegLen + 55} rx={130} ry={18} fill="rgba(26, 26, 46, 0.2)"/>

        {/*
          TWO-LAYER RENDERING:
          Layer 1: ALL fills (no strokes) - creates solid unified shape
          Layer 2: Strokes ONLY on outer boundaries (body, head, hands, feet)

          This eliminates internal stroke boundaries while keeping clean outlines.
        */}

        {/* ========== LAYER 1: ALL FILLS (no strokes) ========== */}

        {/* Joint circles - fill gaps at all connection points */}
        <circle cx={lHip.x} cy={lHip.y} r={legW * 0.6} fill={fill}/>
        <circle cx={rHip.x} cy={rHip.y} r={legW * 0.6} fill={fill}/>
        <circle cx={lKnee.x} cy={lKnee.y} r={legW * 0.6} fill={fill}/>
        <circle cx={rKnee.x} cy={rKnee.y} r={legW * 0.6} fill={fill}/>
        <circle cx={lShoulder.x} cy={lShoulder.y} r={armW * 0.6} fill={fill}/>
        <circle cx={rShoulder.x} cy={rShoulder.y} r={armW * 0.6} fill={fill}/>
        <circle cx={lElbow.x} cy={lElbow.y} r={armW * 0.6} fill={fill}/>
        <circle cx={rElbow.x} cy={rElbow.y} r={armW * 0.6} fill={fill}/>

        {/* Leg segment fills */}
        <path d={limbPath(lHip, lKnee, legW)} fill={fill}/>
        <path d={limbPath(lKnee, lAnkle, legW * 0.9)} fill={fill}/>
        <path d={limbPath(rHip, rKnee, legW)} fill={fill}/>
        <path d={limbPath(rKnee, rAnkle, legW * 0.9)} fill={fill}/>

        {/* Foot fills */}
        <circle cx={lAnkle.x} cy={lAnkle.y} r={footR} fill={fill}/>
        <circle cx={rAnkle.x} cy={rAnkle.y} r={footR} fill={fill}/>

        {/* Body fill */}
        <path d={bodyPath} fill={fill}/>

        {/* Head fill */}
        <circle cx={0} cy={0} r={headR} fill={fill}/>

        {/* Arm segment fills */}
        <path d={limbPath(lShoulder, lElbow, armW)} fill={fill}/>
        <path d={limbPath(lElbow, lWrist, armW * 0.9)} fill={fill}/>
        <path d={limbPath(rShoulder, rElbow, armW)} fill={fill}/>
        <path d={limbPath(rElbow, rWrist, armW * 0.9)} fill={fill}/>

        {/* Hand fills */}
        <circle cx={lWrist.x} cy={lWrist.y} r={handR} fill={fill}/>
        <circle cx={rWrist.x} cy={rWrist.y} r={handR} fill={fill}/>

        {/* ========== LAYER 2: OUTER STROKES ONLY ========== */}

        {/* Body outline */}
        <path d={bodyPath} fill="none" stroke={stroke} strokeWidth={sw}/>

        {/* Head outline + features */}
        <circle cx={0} cy={0} r={headR} fill="none" stroke={stroke} strokeWidth={sw}/>
        <path d="M 30 -75 A 70 70 0 0 1 75 -35" stroke="white" strokeWidth="14" fill="none" strokeLinecap="round" opacity="0.35"/>
        <ellipse cx={-24} cy={8} rx={10} ry={13} fill="#1a1a2e"/>
        <ellipse cx={24} cy={8} rx={10} ry={13} fill="#1a1a2e"/>
        <circle cx={-27} cy={5} r={3} fill="white"/>
        <circle cx={21} cy={5} r={3} fill="white"/>

        {/* Foot outlines (visible endpoints) */}
        <circle cx={lAnkle.x} cy={lAnkle.y} r={footR} fill="none" stroke={stroke} strokeWidth={sw}/>
        <circle cx={rAnkle.x} cy={rAnkle.y} r={footR} fill="none" stroke={stroke} strokeWidth={sw}/>

        {/* Hand outlines (visible endpoints) */}
        <circle cx={lWrist.x} cy={lWrist.y} r={handR} fill="none" stroke={stroke} strokeWidth={sw}/>
        <circle cx={rWrist.x} cy={rWrist.y} r={handR} fill="none" stroke={stroke} strokeWidth={sw}/>
      </g>
    </svg>
  );
};

export const AnimatedUnifiedBoneCharacter: React.FC<{
  poseSequence: {pose: keyof typeof POSES; duration: number}[];
  x?: number;
  y?: number;
  scale?: number;
}> = ({ poseSequence, x = 960, y = 540, scale = 1 }) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  let currentFrame = 0;
  const ranges: {pose: Pose; startFrame: number; endFrame: number}[] = [];
  for (const item of poseSequence) {
    const dur = item.duration * fps;
    ranges.push({ pose: POSES[item.pose], startFrame: currentFrame, endFrame: currentFrame + dur });
    currentFrame += dur;
  }

  let currentPose = POSES.standing;
  for (let i = 0; i < ranges.length; i++) {
    const r = ranges[i];
    if (frame >= r.startFrame && frame < r.endFrame) {
      const next = i < ranges.length - 1 ? ranges[i + 1].pose : r.pose;
      const transDur = 12;
      const transStart = r.endFrame - transDur;
      if (frame >= transStart) {
        const t = (frame - transStart) / transDur;
        const eased = t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t + 2, 2) / 2;
        currentPose = interpolatePose(r.pose, next, eased);
      } else {
        currentPose = r.pose;
      }
      break;
    }
  }

  return <UnifiedBoneCharacter pose={currentPose} x={x} y={y + Math.sin(frame * 0.08) * 2} scale={scale}/>;
};
