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
  standing: { rightShoulder: 12, rightElbow: 15, leftShoulder: 12, leftElbow: 15, rightHip: 3, rightKnee: 0, leftHip: 3, leftKnee: 0 },
  waving: { rightShoulder: 140, rightElbow: 30, leftShoulder: 12, leftElbow: 15, rightHip: 3, rightKnee: 0, leftHip: 3, leftKnee: 0 },
  pointing: { rightShoulder: 85, rightElbow: 0, leftShoulder: 12, leftElbow: 15, rightHip: 3, rightKnee: 0, leftHip: 3, leftKnee: 0 },
  presenting: { rightShoulder: 110, rightElbow: 8, leftShoulder: 12, leftElbow: 15, rightHip: 3, rightKnee: 0, leftHip: 3, leftKnee: 0 },
  thinking: { rightShoulder: 100, rightElbow: 130, leftShoulder: 12, leftElbow: 15, rightHip: 3, rightKnee: 0, leftHip: 3, leftKnee: 0 },
  celebrating: { rightShoulder: 150, rightElbow: 15, leftShoulder: 150, leftElbow: 15, rightHip: 3, rightKnee: 0, leftHip: 3, leftKnee: 0 },
  jumping: { rightShoulder: 130, rightElbow: 8, leftShoulder: 130, leftElbow: 8, rightHip: 20, rightKnee: 30, leftHip: 20, leftKnee: 30 },
  walking1: { rightShoulder: 25, rightElbow: 35, leftShoulder: 5, leftElbow: 8, rightHip: -8, rightKnee: 12, leftHip: 18, leftKnee: 25 },
  walking2: { rightShoulder: 5, rightElbow: 8, leftShoulder: 25, leftElbow: 35, rightHip: 18, rightKnee: 25, leftHip: -8, leftKnee: 12 },
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

// Smooth pill/capsule shape between two points
function capsulePath(start: {x: number, y: number}, end: {x: number, y: number}, startW: number, endW: number): string {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len < 1) return '';

  const nx = -dy / len;
  const ny = dx / len;

  // Rounded ends with smooth bezier connections
  const s1 = { x: start.x + nx * startW/2, y: start.y + ny * startW/2 };
  const s2 = { x: start.x - nx * startW/2, y: start.y - ny * startW/2 };
  const e1 = { x: end.x + nx * endW/2, y: end.y + ny * endW/2 };
  const e2 = { x: end.x - nx * endW/2, y: end.y - ny * endW/2 };

  return `
    M ${s1.x} ${s1.y}
    Q ${start.x + nx * startW/2 + dx * 0.3} ${start.y + ny * startW/2 + dy * 0.3}
      ${e1.x} ${e1.y}
    A ${endW/2} ${endW/2} 0 0 1 ${e2.x} ${e2.y}
    Q ${start.x - nx * startW/2 + dx * 0.3} ${start.y - ny * startW/2 + dy * 0.3}
      ${s2.x} ${s2.y}
    A ${startW/2} ${startW/2} 0 0 1 ${s1.x} ${s1.y}
    Z
  `;
}

interface Props {
  pose: Pose;
  x?: number;
  y?: number;
  scale?: number;
}

export const CelShadedCharacter: React.FC<Props> = ({ pose, x = 960, y = 540, scale = 1 }) => {
  // Dimensions matching golden style
  const headR = 115;
  const bodyW = 120;
  const bodyH = 220;
  const neckY = headR - 15;
  const bodyY = neckY + 25;

  const shoulderX = 50;
  const shoulderY = bodyY + 35;
  const hipX = 38;
  const hipY = bodyY + bodyH - 35;

  const upperArmLen = 95;
  const lowerArmLen = 90;
  const upperArmW = 48;
  const lowerArmW = 42;
  const handR = 38;

  const upperLegLen = 115;
  const lowerLegLen = 105;
  const upperLegW = 55;
  const lowerLegW = 48;
  const footR = 32;

  // Calculate joint positions
  const rShoulder = { x: shoulderX, y: shoulderY };
  const rElbow = pt(rShoulder.x, rShoulder.y, pose.rightShoulder, upperArmLen);
  const rWrist = pt(rElbow.x, rElbow.y, pose.rightShoulder + pose.rightElbow, lowerArmLen);

  const lShoulder = { x: -shoulderX, y: shoulderY };
  const lElbow = pt(lShoulder.x, lShoulder.y, -pose.leftShoulder, upperArmLen);
  const lWrist = pt(lElbow.x, lElbow.y, -pose.leftShoulder - pose.leftElbow, lowerArmLen);

  const rHip = { x: hipX, y: hipY };
  const rKnee = pt(rHip.x, rHip.y, pose.rightHip, upperLegLen);
  const rAnkle = pt(rKnee.x, rKnee.y, pose.rightHip - pose.rightKnee, lowerLegLen);

  const lHip = { x: -hipX, y: hipY };
  const lKnee = pt(lHip.x, lHip.y, -pose.leftHip, upperLegLen);
  const lAnkle = pt(lKnee.x, lKnee.y, -pose.leftHip + pose.leftKnee, lowerLegLen);

  // Body path - smooth oval torso
  const bodyPath = `
    M 0 ${neckY}
    C ${bodyW * 0.5} ${neckY} ${bodyW * 0.95} ${bodyY + 40} ${bodyW * 0.85} ${bodyY + bodyH * 0.5}
    C ${bodyW * 0.75} ${bodyY + bodyH * 0.85} ${bodyW * 0.4} ${bodyY + bodyH} 0 ${bodyY + bodyH}
    C ${-bodyW * 0.4} ${bodyY + bodyH} ${-bodyW * 0.75} ${bodyY + bodyH * 0.85} ${-bodyW * 0.85} ${bodyY + bodyH * 0.5}
    C ${-bodyW * 0.95} ${bodyY + 40} ${-bodyW * 0.5} ${neckY} 0 ${neckY}
    Z
  `;

  const fill = "url(#celGrad)";
  const stroke = "#202038";
  const sw = 11;

  // Highlight paths for cel-shading effect
  const headHighlight = `M ${headR * 0.22} ${-headR * 0.78} A ${headR * 0.7} ${headR * 0.7} 0 0 1 ${headR * 0.7} ${-headR * 0.32}`;

  // Body highlight
  const bodyHighlight = `M ${-bodyW * 0.3} ${bodyY + 20} Q ${-bodyW * 0.5} ${bodyY + bodyH * 0.3} ${-bodyW * 0.35} ${bodyY + bodyH * 0.5}`;

  // Determine z-order based on pose (which limbs are in front)
  const rightArmInFront = pose.rightShoulder > 90;
  const leftArmInFront = pose.leftShoulder > 90;
  const rightLegInFront = pose.rightHip > 10;
  const leftLegInFront = pose.leftHip > 10;

  // Limb rendering helper with highlight
  const renderArm = (shoulder: {x: number, y: number}, elbow: {x: number, y: number}, wrist: {x: number, y: number}, side: 'left' | 'right', id: string) => {
    const mirror = side === 'left' ? -1 : 1;
    return (
      <g key={id}>
        {/* Upper arm */}
        <path d={capsulePath(shoulder, elbow, upperArmW, upperArmW * 0.9)} fill={fill}/>
        {/* Lower arm */}
        <path d={capsulePath(elbow, wrist, lowerArmW, lowerArmW * 0.85)} fill={fill}/>
        {/* Hand */}
        <circle cx={wrist.x} cy={wrist.y} r={handR} fill={fill}/>
        {/* Hand highlight */}
        <circle cx={wrist.x - 8 * mirror} cy={wrist.y - 10} r={handR * 0.3} fill="white" opacity={0.25}/>
        {/* Elbow joint cover */}
        <circle cx={elbow.x} cy={elbow.y} r={upperArmW * 0.45} fill={fill}/>
      </g>
    );
  };

  const renderLeg = (hip: {x: number, y: number}, knee: {x: number, y: number}, ankle: {x: number, y: number}, side: 'left' | 'right', id: string) => {
    const mirror = side === 'left' ? -1 : 1;
    return (
      <g key={id}>
        {/* Upper leg */}
        <path d={capsulePath(hip, knee, upperLegW, upperLegW * 0.9)} fill={fill}/>
        {/* Lower leg */}
        <path d={capsulePath(knee, ankle, lowerLegW, lowerLegW * 0.85)} fill={fill}/>
        {/* Foot */}
        <circle cx={ankle.x} cy={ankle.y} r={footR} fill={fill}/>
        {/* Foot highlight */}
        <circle cx={ankle.x - 6 * mirror} cy={ankle.y - 8} r={footR * 0.25} fill="white" opacity={0.2}/>
        {/* Knee joint cover */}
        <circle cx={knee.x} cy={knee.y} r={upperLegW * 0.45} fill={fill}/>
      </g>
    );
  };

  const groundY = hipY + upperLegLen + lowerLegLen + 85;

  return (
    <svg width={1920} height={1080} style={{ position: 'absolute', left: 0, top: 0 }}>
      <defs>
        {/* Main gradient - subtle radial for 2.5D flat look */}
        <radialGradient id="celGrad" cx="40%" cy="25%" r="80%">
          <stop offset="0%" stopColor="#FFFFFF"/>
          <stop offset="40%" stopColor="#F8F8FC"/>
          <stop offset="100%" stopColor="#E5E5EE"/>
        </radialGradient>

        {/* Drop shadow filter */}
        <filter id="celShadow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur in="SourceAlpha" stdDeviation="10"/>
          <feOffset dx="8" dy="12" result="shadow"/>
          <feFlood floodColor="#1a1a2e" floodOpacity="0.35"/>
          <feComposite in2="shadow" operator="in"/>
          <feMerge>
            <feMergeNode/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>

        {/* Outline effect using morphology */}
        <filter id="outline" x="-20%" y="-20%" width="140%" height="140%">
          <feMorphology in="SourceAlpha" result="dilated" operator="dilate" radius={sw/2}/>
          <feFlood floodColor={stroke} result="color"/>
          <feComposite in="color" in2="dilated" operator="in" result="outline"/>
          <feMerge>
            <feMergeNode in="outline"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>

        {/* Clip for head highlight */}
        <clipPath id="headClip">
          <circle cx={0} cy={0} r={headR}/>
        </clipPath>
      </defs>

      <g transform={`translate(${x}, ${y}) scale(${scale})`}>
        {/* Ground shadow */}
        <ellipse cx={0} cy={groundY} rx={160} ry={22} fill="rgba(30, 30, 50, 0.45)"/>

        {/* Main character group with shadow */}
        <g filter="url(#celShadow)">
          {/* Use outline filter for automatic stroke around all fills */}
          <g filter="url(#outline)">

            {/* === BACK LAYER: Limbs behind body === */}
            {!leftLegInFront && renderLeg(lHip, lKnee, lAnkle, 'left', 'lLegBack')}
            {!rightLegInFront && renderLeg(rHip, rKnee, rAnkle, 'right', 'rLegBack')}
            {!leftArmInFront && renderArm(lShoulder, lElbow, lWrist, 'left', 'lArmBack')}
            {!rightArmInFront && renderArm(rShoulder, rElbow, rWrist, 'right', 'rArmBack')}

            {/* === MIDDLE LAYER: Body === */}
            <path d={bodyPath} fill={fill}/>

            {/* === FRONT LAYER: Limbs in front of body === */}
            {leftLegInFront && renderLeg(lHip, lKnee, lAnkle, 'left', 'lLegFront')}
            {rightLegInFront && renderLeg(rHip, rKnee, rAnkle, 'right', 'rLegFront')}
            {leftArmInFront && renderArm(lShoulder, lElbow, lWrist, 'left', 'lArmFront')}
            {rightArmInFront && renderArm(rShoulder, rElbow, rWrist, 'right', 'rArmFront')}

            {/* === HEAD (always on top) === */}
            <circle cx={0} cy={0} r={headR} fill={fill}/>
          </g>

          {/* === CEL-SHADING HIGHLIGHTS (on top of outline filter) === */}

          {/* Head reflection - curved highlight */}
          <path
            d={headHighlight}
            stroke="#D8D8E0"
            strokeWidth={18}
            fill="none"
            strokeLinecap="round"
            opacity={0.55}
            clipPath="url(#headClip)"
          />

          {/* Body highlight */}
          <path
            d={bodyHighlight}
            stroke="white"
            strokeWidth={14}
            fill="none"
            strokeLinecap="round"
            opacity={0.25}
          />

          {/* === FACE FEATURES === */}
          {/* Eyes */}
          <ellipse cx={-28} cy={5} rx={14} ry={17} fill="#202038"/>
          <ellipse cx={28} cy={5} rx={14} ry={17} fill="#202038"/>

          {/* Eye highlights */}
          <circle cx={-31} cy={0} r={5} fill="#FFFFFF" opacity={0.95}/>
          <circle cx={25} cy={0} r={5} fill="#FFFFFF" opacity={0.95}/>

          {/* Subtle blush for friendly look */}
          <ellipse cx={-55} cy={25} rx={18} ry={10} fill="#FFB5B5" opacity={0.15}/>
          <ellipse cx={55} cy={25} rx={18} ry={10} fill="#FFB5B5" opacity={0.15}/>
        </g>
      </g>
    </svg>
  );
};

export const AnimatedCelShadedCharacter: React.FC<{
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
      const transDur = 18;
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

  // Gentle breathing/bobbing animation
  const breathe = Math.sin(frame * 0.06) * 2;

  return <CelShadedCharacter pose={currentPose} x={x} y={y + breathe} scale={scale}/>;
};
