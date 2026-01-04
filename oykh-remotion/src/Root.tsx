import {Composition} from 'remotion';
import {PoseDemo} from './PoseDemo';
import {BoneDemo} from './BoneDemo';
import {BoneTest} from './BoneTest';
import {GoldenTest} from './GoldenTest';
import {CelTest} from './CelTest';

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="PoseDemo"
        component={PoseDemo}
        durationInFrames={340}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="BoneDemo"
        component={BoneDemo}
        durationInFrames={390}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="BoneTest"
        component={BoneTest}
        durationInFrames={540}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="GoldenTest"
        component={GoldenTest}
        durationInFrames={300}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="CelTest"
        component={CelTest}
        durationInFrames={330}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
