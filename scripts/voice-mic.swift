// Default macOS microphone through VoiceProcessingIO, as mono 16 kHz s16le.
import AudioToolbox
import CoreAudio
import Foundation
import Darwin

func check(_ status: OSStatus, _ operation: String) {
    if status != noErr {
        FileHandle.standardError.write(Data("voice-mic: \(operation): OSStatus \(status)\n".utf8))
        exit(1)
    }
}

var description = AudioComponentDescription(componentType: kAudioUnitType_Output,
    componentSubType: kAudioUnitSubType_VoiceProcessingIO,
    componentManufacturer: kAudioUnitManufacturer_Apple, componentFlags: 0, componentFlagsMask: 0)
guard let component = AudioComponentFindNext(nil, &description) else { exit(1) }
var instance: AudioUnit?
check(AudioComponentInstanceNew(component, &instance), "create VoiceProcessingIO")
let unit = instance!
var enabled: UInt32 = 1
check(AudioUnitSetProperty(unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1,
                          &enabled, 4), "enable input")
check(AudioUnitSetProperty(unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0,
                          &enabled, 4), "enable output")
for (selector, bus) in [(kAudioHardwarePropertyDefaultInputDevice, UInt32(1)),
                         (kAudioHardwarePropertyDefaultOutputDevice, UInt32(0))] {
    var address = AudioObjectPropertyAddress(mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var device: AudioDeviceID = 0
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    check(AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address,
                                    0, nil, &size, &device), "default device")
    check(AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global,
                              bus, &device, size), "select device bus \(bus)")
}
var format = AudioStreamBasicDescription(mSampleRate: 16000,
    mFormatID: kAudioFormatLinearPCM, mFormatFlags: kLinearPCMFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
    mBytesPerPacket: 2, mFramesPerPacket: 1, mBytesPerFrame: 2, mChannelsPerFrame: 1, mBitsPerChannel: 16,
    mReserved: 0)
for (scope, bus) in [(kAudioUnitScope_Output, UInt32(1)), (kAudioUnitScope_Input, UInt32(0))] {
    check(AudioUnitSetProperty(unit, kAudioUnitProperty_StreamFormat, scope, bus, &format,
                              UInt32(MemoryLayout.size(ofValue: format))), "mono client format")
}
var bypass: UInt32 = CommandLine.arguments.contains("--bypass") ? 1 : 0
check(AudioUnitSetProperty(unit, kAUVoiceIOProperty_BypassVoiceProcessing, kAudioUnitScope_Global,
                          0, &bypass, 4), "voice processing bypass")
// Disable automatic gain control so level comparisons measure cancellation rather than gain changes.
var agc: UInt32 = 0
check(AudioUnitSetProperty(unit, kAUVoiceIOProperty_VoiceProcessingEnableAGC, kAudioUnitScope_Global,
                          0, &agc, 4), "disable AGC")
if #available(macOS 14.0, *) {
    var ducking = AUVoiceIOOtherAudioDuckingConfiguration(mEnableAdvancedDucking: false,
        mDuckingLevel: .min)
    check(AudioUnitSetProperty(unit, kAUVoiceIOProperty_OtherAudioDuckingConfiguration,
        kAudioUnitScope_Global, 0, &ducking, UInt32(MemoryLayout.size(ofValue: ducking))), "minimum ducking")
}
// Keep allocation out of the real-time callback.
var maxFrames: UInt32 = 0
var propertySize: UInt32 = 4
check(AudioUnitGetProperty(unit, kAudioUnitProperty_MaximumFramesPerSlice, kAudioUnitScope_Global,
    0, &maxFrames, &propertySize), "maximum callback frames")
let storage = UnsafeMutableRawPointer.allocate(byteCount: Int(maxFrames) * 2, alignment: 16)
setbuf(stdout, nil)
var inputCallback = AURenderCallbackStruct(inputProc: { _, flags, timestamp, _, frames, _ in
    if frames > maxFrames { return kAudio_ParamError }
    let bytes = Int(frames) * 2
    var buffers = AudioBufferList(mNumberBuffers: 1,
        mBuffers: AudioBuffer(mNumberChannels: 1, mDataByteSize: UInt32(bytes), mData: storage))
    let status = AudioUnitRender(unit, flags, timestamp, 1, frames, &buffers)
    if status == noErr { _ = fwrite(storage, 1, Int(buffers.mBuffers.mDataByteSize), stdout) }
    return status
}, inputProcRefCon: nil)
check(AudioUnitSetProperty(unit, kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global,
    1, &inputCallback, UInt32(MemoryLayout.size(ofValue: inputCallback))), "input callback")
var outputCallback = AURenderCallbackStruct(inputProc: { _, flags, _, _, _, buffers in
    flags.pointee.insert(.unitRenderAction_OutputIsSilence)
    if let buffers {
        for buffer in UnsafeMutableAudioBufferListPointer(buffers) {
            if let data = buffer.mData { memset(data, 0, Int(buffer.mDataByteSize)) }
        }
    }
    return noErr
}, inputProcRefCon: nil)
check(AudioUnitSetProperty(unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input,
    0, &outputCallback, UInt32(MemoryLayout.size(ofValue: outputCallback))), "silent output callback")
check(AudioUnitInitialize(unit), "initialize")
check(AudioOutputUnitStart(unit), "start")
FileHandle.standardError.write(Data("voice-mic: mono VoiceProcessingIO active (bypass=\(bypass))\n".utf8))
RunLoop.main.run()
