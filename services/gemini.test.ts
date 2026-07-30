import { generateViralVideoScript, refineScript, fetchSuggestions, fetchHooks } from './gemini';
import { GoogleGenerativeAI } from '@google/generative-ai';

jest.mock('@google/generative-ai', () => {
  const mockGenerateContent = jest.fn().mockResolvedValue({
    response: {
      candidates: [
        {
          content: {
            parts: [{ text: JSON.stringify({ chapters: [] }) }],
          },
        },
      ],
    },
  });

  return {
    GoogleGenerativeAI: jest.fn().mockImplementation(() => ({
      getGenerativeModel: jest.fn(() => ({
        generateContent: mockGenerateContent,
      })),
    })),
    mockGenerateContent, // Export for use in tests
  };
});

describe('Gemini Service', () => {
  const { mockGenerateContent } = require('@google/generative-ai');

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('should generate a viral video script', async () => {
    await generateViralVideoScript('topic', 'hook', 'cosmic');
    expect(mockGenerateContent).toHaveBeenCalled();
  });

  it('should refine a script', async () => {
    await refineScript({} as any, 'feedback');
    expect(mockGenerateContent).toHaveBeenCalled();
  });

  it('should fetch suggestions', async () => {
    await fetchSuggestions();
    expect(mockGenerateContent).toHaveBeenCalled();
  });

  it('should fetch hooks', async () => {
    await fetchHooks('topic');
    expect(mockGenerateContent).toHaveBeenCalled();
  });
});
