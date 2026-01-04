import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {UnifiedBoneCharacter, POSES} from './UnifiedBoneCharacter';

export const BoneTest: React.FC = () => {
  const frame = useCurrentFrame();

  // Cycle through poses every 60 frames (2 seconds at 30fps)
  const poseNames = Object.keys(POSES);
  const poseIndex = Math.floor(frame / 60) % poseNames.length;
  const currentPoseName = poseNames[poseIndex];
  const currentPose = POSES[currentPoseName];

  return (
    <AbsoluteFill style={{
      background: 'linear-gradient(135deg, #45b8d1 0%, #6a6acd 30%, #c080a8 60%, #e8a060 100%)',
    }}>
      <UnifiedBoneCharacter
        pose={currentPose}
        x={960}
        y={500}
        scale={0.9}
      />

      <div style={{
        position: 'absolute',
        bottom: 80,
        width: '100%',
        textAlign: 'center',
        fontSize: 48,
        fontFamily: 'Arial, sans-serif',
        fontWeight: 'bold',
        color: 'white',
        textShadow: '3px 3px 6px rgba(0,0,0,0.3)',
      }}>
        {currentPoseName}
      </div>
    </AbsoluteFill>
  );
};
