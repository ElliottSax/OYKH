import {AbsoluteFill} from 'remotion';
import {AnimatedGoldenCharacter} from './GoldenCharacter';

export const GoldenTest: React.FC = () => {
  return (
    <AbsoluteFill style={{
      background: 'linear-gradient(135deg, #45b8d1 0%, #6a6acd 30%, #c080a8 60%, #e8a060 100%)'
    }}>
      <AnimatedGoldenCharacter
        poseSequence={[
          {pose: 'standing', duration: 1},
          {pose: 'waving', duration: 1},
          {pose: 'pointing', duration: 1},
          {pose: 'presenting', duration: 1},
          {pose: 'thinking', duration: 1},
          {pose: 'celebrating', duration: 1},
          {pose: 'jumping', duration: 1},
          {pose: 'walking1', duration: 0.5},
          {pose: 'walking2', duration: 0.5},
          {pose: 'walking1', duration: 0.5},
          {pose: 'walking2', duration: 0.5},
          {pose: 'standing', duration: 1},
        ]}
        y={500}
        scale={0.9}
      />
    </AbsoluteFill>
  );
};
