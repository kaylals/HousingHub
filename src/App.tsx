import React from 'react';
import './App.css';
import { Grid, Paper, SelectChangeEvent, styled, Button } from '@mui/material';
import CustomFormControl from './component/UI/CustomFormControl';
import TableauEmbed from './component/UI/TableauEmbed';

const StyledContainer = styled(Grid)({
  padding: 16,
  flex: 1,
  overflow: 'auto',
});

const StyledPaper = styled(Paper)({
  padding: 16,
  textAlign: 'center',
  color: 'text.secondary',
  height: '100%'
});

const App: React.FC = () => {
  const [select1, setSelect1] = React.useState('');
  const [select2, setSelect2] = React.useState('');
  const [select3, setSelect3] = React.useState('');

  const handleChange1 = (event: SelectChangeEvent) =>
    setSelect1(event.target.value as string);
  const handleChange2 = (event: SelectChangeEvent) =>
    setSelect2(event.target.value as string);
  const handleChange3 = (event: SelectChangeEvent) =>
    setSelect3(event.target.value as string);

  const regions = [
    { value: "mag", label: 'Magnolia' }
  ];

  return (
    <StyledContainer container spacing={3}>
      <Grid item xs={12}>
        <StyledPaper>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={6} md={3}>
              <CustomFormControl
                label="Option 1"
                value={select1}
                onChange={handleChange1}
                options={regions}
              />
            </Grid>
            <Grid item xs={6} md={3}>
              <CustomFormControl
                label="Option 2"
                value={select2}
                onChange={handleChange2}
                options={regions}
              />
            </Grid>
            <Grid item xs={6} md={3}>
              <CustomFormControl
                label="Option 3"
                value={select3}
                onChange={handleChange3}
                options={regions}
              />
            </Grid>
            <Grid item xs={6} md={3} style={{ textAlign: 'center' }}>
              <Button variant="contained" color="primary" style={{ height: '36px' }}>
                Click Me
              </Button>
            </Grid>
          </Grid>
        </StyledPaper>
      </Grid>
      <Grid item xs={12} style={{ height: '600px', marginTop: '2%' }}>
        <StyledPaper style={{ height: '600px', overflow: 'auto' }}>
          {/* Embed Tableau Dashboard here */}
          <TableauEmbed />
        </StyledPaper>
      </Grid>
    </StyledContainer>
  );
};

export default App;
