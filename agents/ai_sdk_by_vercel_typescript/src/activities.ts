const USER_AGENT = '(temporal-ai-cookbook, cookbook@temporal.io)';

async function geocode(location: string): Promise<{ name: string; latitude: number; longitude: number }> {
  const url = `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(location)}&count=1&countryCode=US`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Geocoding request failed for "${location}": ${response.status}`);
  }
  const data = (await response.json()) as {
    results?: Array<{ name: string; latitude: number; longitude: number }>;
  };
  const match = data.results?.[0];
  if (!match) {
    throw new Error(`No location found for "${location}". The National Weather Service only covers the US.`);
  }
  return match;
}

// The National Weather Service API only covers the US and requires a two-step
// lookup: resolve coordinates to a forecast grid, then fetch that grid's forecast.
export async function getWeather(input: {
  location: string;
}): Promise<{ city: string; temperatureRange: string; conditions: string }> {
  const { name, latitude, longitude } = await geocode(input.location);

  const pointsResponse = await fetch(`https://api.weather.gov/points/${latitude.toFixed(4)},${longitude.toFixed(4)}`, {
    headers: { 'User-Agent': USER_AGENT },
  });
  if (!pointsResponse.ok) {
    throw new Error(`National Weather Service points lookup failed: ${pointsResponse.status}`);
  }
  const points = (await pointsResponse.json()) as { properties: { forecast: string } };

  const forecastResponse = await fetch(points.properties.forecast, { headers: { 'User-Agent': USER_AGENT } });
  if (!forecastResponse.ok) {
    throw new Error(`National Weather Service forecast lookup failed: ${forecastResponse.status}`);
  }
  const forecast = (await forecastResponse.json()) as {
    properties: { periods: Array<{ temperature: number; temperatureUnit: string; shortForecast: string }> };
  };
  const current = forecast.properties.periods[0];

  return {
    city: name,
    temperatureRange: `${current.temperature}${current.temperatureUnit}`,
    conditions: current.shortForecast,
  };
}

export async function calculateAreaOfCircle(input: { radius: number }): Promise<{ area: number }> {
  return { area: Math.PI * input.radius * input.radius };
}
