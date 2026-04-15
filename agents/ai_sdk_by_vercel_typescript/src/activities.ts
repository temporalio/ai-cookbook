export async function getWeather(input: {
  location: string;
}): Promise<{ city: string; temperatureRange: string; conditions: string }> {
  console.log('Activity execution');
  return {
    city: input.location,
    temperatureRange: '14-20C',
    conditions: 'Sunny with wind.',
  };
}

export async function calculateAreaOfCircle(input: {
    radius: number;
  }): Promise<{ area: number }> {
    return { area: Math.PI * input.radius * input.radius };
  }